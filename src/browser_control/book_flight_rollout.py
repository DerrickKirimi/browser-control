"""Token-preserving book-flight rollouts for TRL's custom rollout interface."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from browser_control.actions import InvalidActionError
from browser_control.book_flight import (
    BOOK_FLIGHT_TOOLS,
    initial_messages,
    observation_text,
    parse_book_flight_action,
)
from browser_control.book_flight_sft import TRAINING_SEEDS


@dataclass(frozen=True)
class TokenTurn:
    token_ids: tuple[int, ...]
    sampling_logprobs: tuple[float, ...]
    text: str

    def __post_init__(self):
        if not self.token_ids or len(self.token_ids) != len(self.sampling_logprobs):
            raise ValueError("Generated IDs and logprobs must be nonempty and aligned")
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError("Generated IDs must be nonnegative integers")
        if any(not math.isfinite(p) or p > 1e-6 for p in self.sampling_logprobs):
            raise ValueError("Selected-token sampling logprobs must be finite and <= 0")


class TransformersTokenSampler:
    """Sample from raw history; scores describe the processed sampling distribution.

    TRL's sampling_per_token_logps are distinct from its internally calculated
    old/current policy logprobs. Top-p processing is included in these sampling
    probabilities; do not substitute them for unprocessed policy logprobs.
    """

    def __init__(self, model, tokenizer, torch, decoder):
        if not decoder.do_sample:
            raise ValueError("GRPO token collection requires a sampling decoder")
        self.model, self.tokenizer, self.torch, self.decoder = (
            model,
            tokenizer,
            torch,
            decoder,
        )

    def sample(self, prompt_ids: tuple[int, ...], *, seed: int) -> TokenTurn:
        torch = self.torch
        torch.manual_seed(seed)
        device = next(self.model.parameters()).device
        ids = torch.tensor([prompt_ids], device=device)
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                **self.decoder.generation_kwargs(),
                return_dict_in_generate=True,
                output_scores=True,
            )
        if not torch.equal(generated.sequences[:, : ids.shape[-1]], ids):
            raise ValueError("Generation changed or truncated the conditioning IDs")
        completion = generated.sequences[0, ids.shape[-1] :]
        if len(generated.scores) != len(completion):
            raise ValueError("Generation score count does not match generated IDs")
        logprobs = tuple(
            float(score[0].float().log_softmax(-1)[token].item())
            for score, token in zip(generated.scores, completion, strict=True)
        )
        return TokenTurn(
            tuple(completion.tolist()),
            logprobs,
            self.tokenizer.decode(completion, skip_special_tokens=True),
        )


def chat_ids(tokenizer, messages, *, generation: bool) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tools=BOOK_FLIGHT_TOOLS,
        enable_thinking=False,
        add_generation_prompt=generation,
        tokenize=True,
        return_dict=True,
    )["input_ids"]


def feedback_suffix(tokenizer, action, observation) -> tuple[int, ...]:
    """Render only external feedback; never decode/re-encode sampled tokens.

    A minimal canonical conversation locates the tool-response boundary. The
    pinned template's whitespace after assistant EOS belongs to the external
    suffix, together with the tool response and next assistant header.
    """
    messages = [{"role": "user", "content": "boundary"}, action.message()]
    closed = chat_ids(tokenizer, messages, generation=False)
    continued = chat_ids(
        tokenizer,
        [*messages, {"role": "tool", "content": observation_text(observation)}],
        generation=True,
    )
    if continued[: len(closed)] != closed:
        raise ValueError("Tool feedback rewrote the preceding template prefix")
    eos = tokenizer.eos_token_id
    last_eos = len(closed) - 1 - closed[::-1].index(eos)
    tail = closed[last_eos + 1 :]
    if tokenizer.decode(tail, skip_special_tokens=False).strip():
        raise ValueError("Expected only template whitespace after assistant EOS")
    return tuple([*tail, *continued[len(closed) :]])


def collect_token_episode(
    environment: Any,
    sampler: Any,
    tokenizer: Any,
    task_seed: int,
    *,
    generation_seed: int,
    max_steps: int = 15,
    max_prompt_tokens: int = 16384,
    training: bool = True,
) -> dict:
    """Strict native task semantics with raw IDs and external-token loss masks."""
    if training and task_seed not in TRAINING_SEEDS:
        raise ValueError("Training rollouts accept only seeds 100..9999")
    if not 1 <= max_steps <= 15:
        raise ValueError("max_steps must be in 1..15")
    state = environment.reset(seed=task_seed)
    prompt = tuple(
        chat_ids(tokenizer, initial_messages(state.observation), generation=True)
    )
    completion: list[int] = []
    logprobs: list[float] = []
    mask: list[int] = []
    steps = []
    stop = "horizon"
    for turn in range(max_steps):
        context = (*prompt, *completion)
        if len(context) > max_prompt_tokens:
            stop = "prompt-budget"
            break
        output = sampler.sample(context, seed=generation_seed + turn)
        completion.extend(output.token_ids)
        logprobs.extend(output.sampling_logprobs)
        mask.extend([1] * len(output.token_ids))
        row = {
            "turn": turn,
            "text": output.text,
            "token_ids": list(output.token_ids),
            "sampling_logprobs": list(output.sampling_logprobs),
            "generation_seed": generation_seed + turn,
            "prompt_tokens": len(context),
            "observation": asdict(state.observation),
        }
        steps.append(row)
        try:
            action = parse_book_flight_action(output.text)
        except InvalidActionError as error:
            row.update(valid_action=False, error=str(error))
            stop = "invalid-action"
            break
        row.update(valid_action=True, action=action.render())
        state = environment.step(action.render())
        row.update(
            action_error=state.observation.last_action_error,
            reward=state.reward,
            done=state.done,
        )
        if state.done:
            stop = "success" if state.reward > 0 else "terminal-failure"
            break
        if turn + 1 < max_steps:
            # A valid cap-length call may lack EOS. The forced message separator
            # is context, not a sampled token; it receives mask 0 and logprob 0.
            suffix = feedback_suffix(tokenizer, action, state.observation)
            if completion[-1] != tokenizer.eos_token_id:
                suffix = (tokenizer.eos_token_id, *suffix)
            completion.extend(suffix)
            logprobs.extend([0.0] * len(suffix))
            mask.extend([0] * len(suffix))
    if not completion or not any(mask):
        raise ValueError("Rollout produced no model tokens within its prompt budget")
    return {
        "task_seed": task_seed,
        "goal": state.observation.goal,
        "prompt_ids": list(prompt),
        "completion_ids": completion,
        "logprobs": logprobs,
        "env_mask": mask,
        "native_success": bool(state.done and state.reward > 0),
        "success": bool(state.done and state.reward > 0),
        "native_reward": state.reward,
        "environment_steps": sum(bool(step.get("valid_action")) for step in steps),
        "action_errors": sum(bool(step.get("action_error")) for step in steps),
        "final_observation": asdict(state.observation),
        "stop_reason": stop,
        "steps": steps,
    }


def native_success_reward(completions, native_success, **kwargs) -> list[float]:
    """Native positive terminal reward is the only sparse success signal."""
    if len(completions) != len(native_success):
        raise ValueError("Native success rows must align with completions")
    if any(type(success) is not bool for success in native_success):
        raise ValueError("Expected boolean native terminal successes")
    return [float(success) for success in native_success]


class BookFlightRollout:
    """Single-process TRL rollout_func; dataset strings carry reset metadata only.

    The actual model prompt comes from env.reset, not the dataset metadata.
    The callback returns its actual prompt IDs, raw generated IDs, sampling
    probabilities and model/external token masks without re-tokenization.
    """

    def __init__(self, environment, tokenizer, sampler_factory):
        self.environment = environment
        self.tokenizer = tokenizer
        self.sampler_factory = sampler_factory
        self.episodes: list[dict] = []

    def __call__(self, prompts, trainer) -> dict:
        if trainer.tools:
            raise ValueError(
                "Custom browser rollouts require trainer.tools to be empty"
            )
        if trainer.accelerator.num_processes != 1:
            raise ValueError("Browser rollout backend currently supports one process")
        if getattr(trainer, "mask_truncated_completions", False):
            raise ValueError("Keep truncated/invalid model tokens in the loss mask")
        group_size = trainer.num_generations
        if not prompts or len(prompts) % group_size:
            raise ValueError("Rollout batch must contain complete generation groups")
        requests = [json.loads(prompt) for prompt in prompts]
        if any(set(request) != {"task_seed"} for request in requests):
            raise ValueError("Dataset prompts must carry only task_seed reset metadata")
        for start in range(0, len(requests), group_size):
            if len({r["task_seed"] for r in requests[start : start + group_size]}) != 1:
                raise ValueError("Each generation group must reset the same task seed")
        model = trainer.accelerator.unwrap_model(trainer.model)
        training = model.training
        model.eval()
        sampler = self.sampler_factory(model)
        rows = []
        try:
            for index, request in enumerate(requests):
                row = collect_token_episode(
                    self.environment,
                    sampler,
                    self.tokenizer,
                    request["task_seed"],
                    generation_seed=1_000_000
                    + trainer.state.global_step * 100_000
                    + index * 16,
                )
                if index % group_size and row["prompt_ids"] != rows[-1]["prompt_ids"]:
                    raise ValueError(
                        "Grouped environment resets produced different prompts"
                    )
                rows.append(row)
        finally:
            model.train(training)
        self.episodes.extend(rows)
        return {
            key: [row[key] for row in rows]
            for key in (
                "prompt_ids",
                "completion_ids",
                "logprobs",
                "env_mask",
                "native_success",
            )
        }
