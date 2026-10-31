"""Exercise real TRL 1.16 batching, env masks and updates with a tiny CPU fixture.

The fixture scripts actions and native outcomes. This validates trainer wiring
and optionally the pinned LFM3 architecture with tiny random weights, not the
pretrained policy, real browser sampling, T4 memory or task learning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import MethodType

from browser_control.book_flight_rollout import (
    BookFlightRollout,
    TokenTurn,
    TransformersTokenSampler,
    native_success_reward,
)
from browser_control.environment import BrowserObservation, BrowserStep
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION
from browser_control.lfm3_trainer import low_scale_grpo_trainer_class
from browser_control.transformers_policy import DecoderConfig

CALL = "<tool_call><function=click><parameter=bid>1</parameter></function></tool_call>"


class FixtureTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        if messages[-1]["role"] == "tool":
            ids = [10, 11, 42, 99, 7, 55, 56, 77]
        elif not add_generation_prompt:
            ids = [10, 11, 42, 99, 7]
        else:
            ids = [10, 11]
        return {"input_ids": ids}

    def decode(self, tokens, **kwargs):
        if tokens != [7]:
            raise ValueError("Fixture only decodes template whitespace")
        return "\n"


class FixtureEnvironment:
    def __init__(self):
        self.resets = 0

    def reset(self, *, seed):
        self.resets += 1
        self.turn = 0
        self.success = self.resets % 2 == 1
        return BrowserStep(BrowserObservation("fixture", "[1] button"), 0.0, False)

    def step(self, action):
        if action != "click('1')" or self.turn >= 2:
            raise ValueError("Invalid fixture execution or post-terminal step")
        self.turn += 1
        return BrowserStep(
            BrowserObservation(
                "fixture", "[1] button", "fixture error" if self.turn == 1 else None
            ),
            float(self.success) if self.turn == 2 else 0.0,
            self.turn == 2,
        )


class FixtureSampler:
    def sample(self, prompt_ids, *, seed):
        ids = (
            (81, 99)
            if len(prompt_ids) == 2
            else ((82, 99) if (seed // 16) % 2 == 0 else (83, 84, 99))
        )
        # Finite aligned fixture values, not estimates of a sampling density.
        return TokenTurn(ids, (-2.0,) * len(ids), CALL)


def main():
    import torch
    import trl
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        GenerationConfig,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--architecture", choices=("llama", "lfm3"), default="llama")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier contract evidence")
    if trl.__version__ != "1.16.0":
        raise ValueError(f"Expected TRL 1.16.0, got {trl.__version__}")
    torch.set_num_threads(2)
    torch.manual_seed(23)
    raw = Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="t0", bos_token="t1", eos_token="t99"
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=99,
        )
    )
    if args.architecture == "lfm3":
        from transformers import AutoConfig, AutoModelForCausalLM

        tiny_config = AutoConfig.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["conv", "full_attention", "sliding_attention"],
            max_position_embeddings=256,
            sliding_window=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=99,
        )
        model = AutoModelForCausalLM.from_config(
            tiny_config,
            trust_remote_code=True,
            code_revision=MODEL_REVISION,
            attn_implementation="sdpa",
            dtype=torch.float32,
        )
    model.generation_config = GenerationConfig(
        pad_token_id=0, eos_token_id=99, bos_token_id=1, top_k=0
    )
    if args.lora:
        from peft import LoraConfig, get_peft_model

        targets = ["q_proj", "v_proj"]
        if args.architecture == "lfm3":
            from preflight_lfm3_attention import LORA

            targets = LORA["target_modules"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=targets,
                task_type="CAUSAL_LM",
            ),
        )
        if any(
            "lora_" not in n for n, p in model.named_parameters() if p.requires_grad
        ):
            raise ValueError("CPU adapter check must train only LoRA parameters")
    # Separately exercise the actual score-producing sampler on real model output.
    sampler = TransformersTokenSampler(
        model.eval(),
        tokenizer,
        torch,
        DecoderConfig("contract", True, 8, temperature=1.0, top_p=0.95),
    )
    sampled = sampler.sample((10, 11), seed=23)
    # Independently reconstruct top-p probabilities from full-prefix forwards.
    # These are sampling probabilities, not TRL's unprocessed policy logprobs.
    context = [10, 11]
    errors = []
    with torch.inference_mode():
        for token, observed in zip(
            sampled.token_ids, sampled.sampling_logprobs, strict=True
        ):
            ids = torch.tensor([context])
            logits = (
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
                .logits[0, -1]
                .float()
            )
            sorted_logits, indices = logits.sort()
            remove = sorted_logits.softmax(-1).cumsum(-1) <= 0.05
            remove[-1] = False
            logits[indices[remove]] = -torch.inf
            expected = logits.log_softmax(-1)[token].item()
            errors.append(abs(expected - observed))
            context.append(token)
    if max(errors) > 1e-5:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "check": "independent-sampling-logprobs",
                    "architecture": args.architecture,
                    "model_revision": MODEL_REVISION,
                    "scope": "tiny random CPU model; not pretrained LFM3/browser training",
                    "absolute_errors": errors,
                    "sampled_ids": sampled.token_ids,
                    "model_config": model.config.to_dict(),
                },
                indent=2,
            )
            + "\n"
        )
        raise ValueError(
            "Recorded sampling probabilities do not match processed top-p scores: "
            f"per-token absolute errors={errors}"
        )
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rollout = BookFlightRollout(
        FixtureEnvironment(), FixtureTokenizer(), lambda m: FixtureSampler()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = trl.GRPOConfig(
        output_dir=str(args.output.parent / (args.output.stem + "-trainer")),
        use_cpu=True,
        bf16=False,
        fp16=False,
        use_vllm=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        steps_per_generation=1,
        num_iterations=1,
        max_steps=2,
        max_completion_length=64,
        learning_rate=1e-3,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        mask_truncated_completions=False,
        disable_dropout=True,
        temperature=1.0,
        top_p=0.95,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        disable_tqdm=True,
        seed=23,
    )
    trainer = low_scale_grpo_trainer_class()(
        model=model,
        args=config,
        processing_class=tokenizer,
        reward_funcs=native_success_reward,
        train_dataset=Dataset.from_list(
            [{"prompt": json.dumps({"task_seed": seed})} for seed in (100, 101)]
        ),
        rollout_func=rollout,
    )
    if trainer.accelerator.scaler_handler.init_scale != 1.0:
        raise ValueError("Explicit initial loss scale did not reach Accelerate")
    # TRL patches the model with its fused LM head. Independently compare that
    # path with the original causal forward on a multi-turn fixture prefix.
    fixture_ids = torch.tensor([[10, 11, 81, 99, 7, 55, 56, 77, 82, 99]])
    with torch.inference_mode():
        fused, _, _ = trainer._get_per_token_logps_and_entropies(
            model, fixture_ids, torch.ones_like(fixture_ids), 8
        )
        logits = (
            model(
                input_ids=fixture_ids,
                attention_mask=torch.ones_like(fixture_ids),
                use_cache=False,
            )
            .logits[:, 1:-1]
            .float()
        )
        expected = (
            logits.log_softmax(-1)
            .gather(-1, fixture_ids[:, 2:].unsqueeze(-1))
            .squeeze(-1)
        )
        fused_error = (fused - expected).abs().max().item()
    if fused_error > 1e-5:
        raise ValueError(
            f"Fused head differs from original policy logprobs: {fused_error}"
        )
    checks = []
    original_loss = trainer._compute_loss
    original_logps = trainer._get_per_token_logps_and_entropies

    def checked_loss(self, current_model, inputs):
        mask = inputs["completion_mask"] * inputs["tool_mask"]
        if not torch.any(mask == 0) or not torch.any(mask == 1):
            raise ValueError("Fixture lost model/external mask distinction")
        if not torch.all(torch.isfinite(inputs["advantages"])) or not torch.any(
            inputs["advantages"] != 0
        ):
            raise ValueError("Fixture native rewards did not produce useful advantages")
        self._contract_active_mask = mask
        checks.append(
            {
                "advantages": inputs["advantages"].tolist(),
                "completion_ids": inputs["completion_ids"].tolist(),
                "tool_mask": inputs["tool_mask"].tolist(),
            }
        )
        return original_loss(current_model, inputs)

    def checked_logps(self, *positional, **kwargs):
        result = original_logps(*positional, **kwargs)
        if result[0].requires_grad:
            active = self._contract_active_mask

            def check_gradient(gradient):
                if not torch.isfinite(gradient).all():
                    raise ValueError("Nonfinite policy-logprob gradient")
                if torch.any(gradient[active == 0] != 0):
                    raise ValueError("External/padding tokens received policy gradient")
                if not torch.any(gradient[active == 1] != 0):
                    raise ValueError("No model token received policy gradient")
                checks[-1]["external_gradients_zero"] = True

            result[0].register_hook(check_gradient)
        return result

    trainer._compute_loss = MethodType(checked_loss, trainer)
    trainer._get_per_token_logps_and_entropies = MethodType(checked_logps, trainer)
    trainer.train()
    changed = sum(
        not torch.equal(p.detach(), before[n]) for n, p in model.named_parameters()
    )
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite parameter after CPU integration updates")
    if any(
        not torch.equal(p.detach(), before[n])
        for n, p in model.named_parameters()
        if not p.requires_grad
    ):
        raise ValueError("Frozen model parameters changed")
    if trainer.state.global_step != 2 or len(checks) != 2 or not changed:
        raise ValueError("Expected two genuine trainer optimizer updates")
    if any(not row.get("external_gradients_zero") for row in checks):
        raise ValueError("Gradient hook did not validate every loss")
    if len(rollout.episodes) != 4:
        raise ValueError("Trainer did not request two complete generation groups")
    for row in checks:
        for ids in row["completion_ids"]:
            unpadded = list(ids)
            while unpadded[-1] == tokenizer.pad_token_id:
                unpadded.pop()
            if unpadded not in [
                episode["completion_ids"] for episode in rollout.episodes
            ]:
                raise ValueError("Trainer rewrote generated/external IDs")
    report = {
        "status": "passed",
        "scope": "tiny random CPU model and scripted environment/actions; not pretrained LFM3/browser training",
        "architecture": args.architecture,
        "model_revision": MODEL_REVISION if args.architecture == "lfm3" else None,
        "model_config": model.config.to_dict(),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "lora_target_modules": targets if args.lora else None,
        "model_source_sha256": hashlib.sha256(
            Path(
                inspect.getfile(type(model.get_base_model() if args.lora else model))
            ).read_bytes()
        ).hexdigest(),
        "trainer_steps": trainer.state.global_step,
        "changed_parameter_tensors": changed,
        "adapter_only": args.lora,
        "configured_initial_scale": trainer.accelerator.scaler_handler.init_scale,
        "fp16_gpu_scaling_executed": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "accelerate", "datasets")
        },
        "trainer_sha256": hashlib.sha256(
            Path(inspect.getfile(trl.GRPOTrainer)).read_bytes()
        ).hexdigest(),
        "sampler_tokens": len(sampled.token_ids),
        "sampling_logprob_max_error": max(errors),
        "fused_policy_logprob_max_error": fused_error,
        "checks": checks,
        "episodes": rollout.episodes,
    }
    if args.lora:
        report["packages"]["peft"] = importlib.metadata.version("peft")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        "Passed: real TRL custom rollout, grouped rewards, external masks, gradients and two updates"
    )


if __name__ == "__main__":
    main()
