"""One-step rollout collection independent of the model implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from browser_control.actions import InvalidActionError, parse_click_test_action
from browser_control.environment import BrowserEnvironment
from browser_control.prompts import build_click_test_messages
from browser_control.trajectory import OneStepTrajectory, TrajectoryGroup


@dataclass(frozen=True, slots=True)
class SampledCompletion:
    """Token-aligned output captured from the behavior policy."""

    text: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    trainable_mask: tuple[bool, ...]


class TokenPolicy(Protocol):
    def sample(
        self, messages: list[dict[str, str]], *, seed: int
    ) -> SampledCompletion: ...


def collect_click_test_group(
    environment_factory: Callable[[], BrowserEnvironment],
    policy: TokenPolicy,
    *,
    task_seed: int,
    generation_seeds: Iterable[int],
) -> TrajectoryGroup:
    """Collect independent completions from repeated resets of one seeded task."""

    members = tuple(
        _collect_member(environment_factory, policy, task_seed, generation_seed)
        for generation_seed in generation_seeds
    )
    return TrajectoryGroup(members)


def _collect_member(
    environment_factory: Callable[[], BrowserEnvironment],
    policy: TokenPolicy,
    task_seed: int,
    generation_seed: int,
) -> OneStepTrajectory:
    environment = environment_factory()
    try:
        initial = environment.reset(seed=task_seed)
        sample = policy.sample(
            build_click_test_messages(initial.observation), seed=generation_seed
        )
        common = {
            "task_id": f"click-test:{task_seed}",
            "prompt_ids": sample.prompt_ids,
            "completion_ids": sample.completion_ids,
            "behavior_logprobs": sample.behavior_logprobs,
            "trainable_mask": sample.trainable_mask,
        }
        try:
            action = parse_click_test_action(sample.text).render()
        except InvalidActionError:
            return OneStepTrajectory(
                **common,
                reward=0.0,
                valid_action=False,
                terminal=True,
            )

        step = environment.step(action)
        return OneStepTrajectory(
            **common,
            reward=step.reward,
            valid_action=True,
            terminal=step.done,
            truncated=not step.done,
        )
    finally:
        environment.close()
