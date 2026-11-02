"""Raw generated IDs survive tool feedback, invalid output and native termination."""

import json
from types import SimpleNamespace

import pytest

from browser_control.book_flight_rollout import (
    BookFlightRollout,
    TokenTurn,
    collect_token_episode,
    native_success_reward,
)
from browser_control.environment import BrowserObservation, BrowserStep

CALL = "<tool_call><function=click><parameter=bid>1</parameter></function></tool_call>"


class Tokenizer:
    eos_token_id = 99

    def __init__(self):
        self.feedback = []

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        assert kwargs["return_dict"] and not kwargs["enable_thinking"]
        if messages[-1]["role"] == "tool":
            self.feedback.append(messages[-1]["content"])
            ids = [10, 11, 42, 99, 7, 55, 56, 77]
        elif not add_generation_prompt:
            ids = [10, 11, 42, 99, 7]
        else:
            ids = [10, 11]
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        assert ids == [7]
        return "\n"


class Environment:
    def __init__(self, done_after=2, reward=1.0):
        self.count, self.done_after, self.reward = 0, done_after, reward

    def reset(self, *, seed):
        self.seed, self.count = seed, 0
        return BrowserStep(BrowserObservation("goal", "[1] button"), 0.0, False)

    def step(self, action):
        assert action == "click('1')"
        self.count += 1
        assert self.count <= self.done_after
        done = self.count == self.done_after
        return BrowserStep(
            BrowserObservation("goal", "[1] button", "readonly"),
            self.reward if done else 0.0,
            done,
        )


class Sampler:
    def __init__(self, turns):
        self.turns, self.contexts, self.seeds = iter(turns), [], []

    def sample(self, prompt_ids, *, seed):
        self.contexts.append(prompt_ids)
        self.seeds.append(seed)
        return next(self.turns)


def test_external_tokens_are_context_and_have_zero_loss_mask():
    tokenizer = Tokenizer()
    # Deliberately noncanonical IDs: re-tokenizing CALL would produce 42.
    sampler = Sampler(
        [
            TokenTurn((81, 82, 99), (-2.0, -3.0, -1.0), CALL),
            TokenTurn((83, 99), (-4.0, -1.0), CALL),
        ]
    )
    env = Environment()
    episode = collect_token_episode(env, sampler, tokenizer, 100, generation_seed=900)
    assert episode["completion_ids"] == [81, 82, 99, 7, 55, 56, 77, 83, 99]
    assert episode["env_mask"] == [1, 1, 1, 0, 0, 0, 0, 1, 1]
    assert episode["logprobs"] == [-2, -3, -1, 0, 0, 0, 0, -4, -1]
    assert sampler.contexts[1] == (10, 11, 81, 82, 99, 7, 55, 56, 77)
    assert sampler.seeds == [900, 901]
    assert "Last action error: readonly" in tokenizer.feedback[0]
    assert episode["native_success"] and episode["stop_reason"] == "success"
    assert env.count == 2


def test_invalid_output_is_retained_and_never_executed():
    env = Environment()
    episode = collect_token_episode(
        env,
        Sampler([TokenTurn((81, 99), (-2, -1), "prose " + CALL)]),
        Tokenizer(),
        100,
        generation_seed=1,
    )
    assert episode["completion_ids"] == [81, 99]
    assert episode["env_mask"] == [1, 1] and env.count == 0
    assert episode["stop_reason"] == "invalid-action"
    assert not episode["native_success"]


def test_missing_generated_eos_is_added_only_as_external_separator():
    sampler = Sampler(
        [TokenTurn((81,), (-2,), CALL), TokenTurn((82, 99), (-2, -1), CALL)]
    )
    episode = collect_token_episode(
        Environment(), sampler, Tokenizer(), 100, generation_seed=1
    )
    assert episode["completion_ids"][:6] == [81, 99, 7, 55, 56, 77]
    assert episode["env_mask"][:6] == [1, 0, 0, 0, 0, 0]


def test_horizon_adds_no_unused_tool_feedback():
    tokenizer, env = Tokenizer(), Environment(done_after=16)
    sampler = Sampler([TokenTurn((81, 99), (-2, -1), CALL)] * 15)
    episode = collect_token_episode(env, sampler, tokenizer, 100, generation_seed=1)
    assert episode["stop_reason"] == "horizon" and env.count == 15
    assert len(tokenizer.feedback) == 14
    assert episode["env_mask"][-2:] == [1, 1]


@pytest.mark.parametrize("seed", [30, 99, 10000, 10100])
def test_training_seed_leak_is_rejected_before_reset(seed):
    with pytest.raises(ValueError, match="Training rollouts"):
        collect_token_episode(None, None, None, seed, generation_seed=1)


@pytest.mark.parametrize(
    "ids,logps", [([], []), ([1], []), ([1], [float("nan")]), ([1], [0.2])]
)
def test_invalid_token_logprob_alignment_is_rejected(ids, logps):
    with pytest.raises(ValueError):
        TokenTurn(tuple(ids), tuple(logps), CALL)


class Model:
    training = True

    def eval(self):
        self.training = False

    def train(self, training=True):
        self.training = training


def trainer():
    return SimpleNamespace(
        model=Model(),
        tools=[],
        num_generations=2,
        state=SimpleNamespace(global_step=0),
        accelerator=SimpleNamespace(num_processes=1, unwrap_model=lambda model: model),
    )


def test_callback_preserves_groups_and_restores_training_mode():
    current = trainer()
    sampler = Sampler([TokenTurn((81, 99), (-2, -1), CALL)] * 4)
    rollout = BookFlightRollout(Environment(), Tokenizer(), lambda model: sampler)
    output = rollout([json.dumps({"task_seed": 100})] * 2, current)
    assert current.model.training
    assert output["prompt_ids"] == [[10, 11], [10, 11]]
    assert output["native_success"] == [True, True]
    assert sampler.seeds == [1_000_000, 1_000_001, 1_000_016, 1_000_017]
    assert native_success_reward(["a", "b"], output["native_success"]) == [1.0, 1.0]


def test_callback_rejects_mixed_seed_generation_group():
    rollout = BookFlightRollout(None, None, None)
    with pytest.raises(ValueError, match="same task seed"):
        rollout([json.dumps({"task_seed": seed}) for seed in (100, 101)], trainer())


def test_callback_can_repeat_the_frozen_diagnostic_sampling_seeds():
    current = trainer()
    sampler = Sampler([TokenTurn((81, 99), (-2, -1), CALL)] * 4)
    rollout = BookFlightRollout(
        Environment(),
        Tokenizer(),
        lambda model: sampler,
        generation_seed_base=1_000_000 + 100 * 64,
    )
    rollout([json.dumps({"task_seed": 100})] * 2, current)
    assert sampler.seeds == [1_006_400, 1_006_401, 1_006_416, 1_006_417]


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_generation_seed_base_is_rejected(seed):
    with pytest.raises(ValueError, match="seed base"):
        BookFlightRollout(None, None, None, generation_seed_base=seed)


def test_callback_rejects_trl_tool_loop_to_avoid_double_execution():
    current = trainer()
    current.tools = ["click"]
    with pytest.raises(ValueError, match="tools to be empty"):
        BookFlightRollout(None, None, None)([], current)


def test_callback_restores_training_mode_on_sampler_error():
    current = trainer()
    with pytest.raises(StopIteration):
        BookFlightRollout(Environment(), Tokenizer(), lambda model: Sampler([]))(
            [json.dumps({"task_seed": 100})] * 2, current
        )
    assert current.model.training
