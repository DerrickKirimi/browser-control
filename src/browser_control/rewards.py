"""Programmatic BrowserGym rewards for one-step click-test training."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from browser_control.actions import InvalidActionError, parse_click_test_action
from browser_control.environment import BrowserEnvironment
from browser_control.prompts import build_click_test_messages

EnvironmentFactory = Callable[[], BrowserEnvironment]


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if (
        isinstance(completion, list)
        and len(completion) == 1
        and isinstance(completion[0], dict)
        and completion[0].get("role") == "assistant"
        and isinstance(completion[0].get("content"), str)
    ):
        return completion[0]["content"]
    raise TypeError("completion must be plain text or one assistant message")


def build_click_test_rows(
    environment_factory: EnvironmentFactory, seeds: Iterable[int]
) -> list[dict[str, Any]]:
    """Freeze seeded initial observations into prompt-only training rows."""

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        environment = environment_factory()
        try:
            initial = environment.reset(seed=seed)
            rows.append(
                {
                    "prompt": build_click_test_messages(initial.observation),
                    "task_seed": seed,
                }
            )
        finally:
            environment.close()
    if not rows:
        raise ValueError("training rows require at least one task seed")
    return rows


def make_click_test_reward(environment_factory: EnvironmentFactory):
    """Return a TRL-compatible reward function backed by BrowserGym."""

    def click_test_reward(
        completions: list[Any], task_seed: list[int], **kwargs: Any
    ) -> list[float]:
        del kwargs
        if len(completions) != len(task_seed):
            raise ValueError("completions and task_seed must have equal lengths")
        rewards: list[float] = []
        for completion, seed in zip(completions, task_seed, strict=True):
            completion_text = _completion_text(completion)
            try:
                action = parse_click_test_action(completion_text).render()
            except InvalidActionError:
                rewards.append(0.0)
                continue

            environment = environment_factory()
            try:
                environment.reset(seed=int(seed))
                result = environment.step(action)
                rewards.append(1.0 if result.done and result.reward > 0 else 0.0)
            finally:
                environment.close()
        return rewards

    return click_test_reward
