from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from browser_control.environment import BrowserStep
from browser_control.rollout import SampledCompletion, collect_click_test_group
from browser_control.testing import FakeClickTestEnvironment


@dataclass
class SeededTokenPolicy:
    outputs: dict[int, str]
    seen_messages: list[list[dict[str, str]]] = field(default_factory=list)

    def sample(
        self, messages: list[dict[str, str]], *, seed: int
    ) -> SampledCompletion:
        self.seen_messages.append(messages)
        completion = (seed, seed + 1, 0)
        return SampledCompletion(
            text=self.outputs[seed],
            prompt_ids=(10, 11, 12),
            completion_ids=completion,
            behavior_logprobs=(-0.1, -0.2, 0.0),
            trainable_mask=(True, True, False),
        )


def test_collection_preserves_success_failure_and_invalid_completions() -> None:
    created: list[FakeClickTestEnvironment] = []

    def factory() -> FakeClickTestEnvironment:
        environment = FakeClickTestEnvironment()
        created.append(environment)
        return environment

    policy = SeededTokenPolicy(
        {1: "click('13')", 2: "noop()", 3: "click(<button id='13'>)"}
    )
    group = collect_click_test_group(
        factory, policy, task_seed=23, generation_seeds=(1, 2, 3)
    )

    success, noop, invalid = group.members
    assert (success.reward, success.terminal, success.truncated) == (1.0, True, False)
    assert (noop.reward, noop.terminal, noop.truncated) == (0.0, True, False)
    assert (invalid.reward, invalid.valid_action, invalid.terminal) == (0.0, False, True)
    assert invalid.completion_ids == (3, 4, 0)
    assert invalid.behavior_logprobs == (-0.1, -0.2, 0.0)
    assert [len(environment.actions) for environment in created] == [1, 1, 0]
    assert all(not environment._active for environment in created)
    assert all(messages[-1]["content"].endswith("Action:") for messages in policy.seen_messages)


def test_valid_nonterminal_action_is_marked_truncated() -> None:
    class NonterminalNoopEnvironment(FakeClickTestEnvironment):
        def step(self, action: str) -> BrowserStep:
            if action != "noop()":
                return super().step(action)
            self.actions.append(action)
            return BrowserStep(
                observation=self._observation(),
                reward=0.0,
                terminated=False,
                info={"task": "click-test", "backend": "fake"},
            )

    group = collect_click_test_group(
        lambda: NonterminalNoopEnvironment(),
        SeededTokenPolicy({1: "noop()", 2: "noop()"}),
        task_seed=23,
        generation_seeds=(1, 2),
    )

    assert all(member.valid_action for member in group.members)
    assert all(not member.terminal and member.truncated for member in group.members)


def test_collection_rejects_too_few_generations() -> None:
    with pytest.raises(ValueError, match="at least two"):
        collect_click_test_group(
            lambda: FakeClickTestEnvironment(),
            SeededTokenPolicy({1: "click('13')"}),
            task_seed=23,
            generation_seeds=(1,),
        )


def test_group_rejects_policy_prompt_tokenization_drift() -> None:
    @dataclass
    class DriftingPolicy(SeededTokenPolicy):
        def sample(
            self, messages: list[dict[str, str]], *, seed: int
        ) -> SampledCompletion:
            sample = super().sample(messages, seed=seed)
            return SampledCompletion(
                text=sample.text,
                prompt_ids=(10, seed),
                completion_ids=sample.completion_ids,
                behavior_logprobs=sample.behavior_logprobs,
                trainable_mask=sample.trainable_mask,
            )

    with pytest.raises(ValueError, match="prompt_ids"):
        collect_click_test_group(
            lambda: FakeClickTestEnvironment(),
            DriftingPolicy({1: "noop()", 2: "noop()"}),
            task_seed=23,
            generation_seeds=(1, 2),
        )
