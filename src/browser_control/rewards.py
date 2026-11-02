"""Programmatic BrowserGym rewards for one-step click-test training."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from difflib import SequenceMatcher
from typing import Any

from browser_control.actions import InvalidActionError, parse_click_test_action
from browser_control.environment import BrowserEnvironment
from browser_control.prompts import build_click_test_messages

EnvironmentFactory = Callable[[], BrowserEnvironment]
_BUTTON_ID = re.compile(r"\[(?:bid=)?([^\]]+)\].*button.*click me", re.IGNORECASE)


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


def _target_action(accessibility_tree: str) -> str:
    for line in accessibility_tree.splitlines():
        match = _BUTTON_ID.search(line)
        if match:
            return f"click({match.group(1).strip()!r})"
    raise RuntimeError("click-test observation has no identifiable target button")


class ClickTestReward:
    """TRL reward callable with auditable per-completion records."""

    __name__ = "click_test_reward"

    def __init__(
        self,
        environment_factory: EnvironmentFactory,
        *,
        format_curriculum: bool,
        curriculum_steps: int | None,
    ) -> None:
        if curriculum_steps is not None and curriculum_steps <= 0:
            raise ValueError("curriculum_steps must be positive when provided")
        self.environment_factory = environment_factory
        self.format_curriculum = format_curriculum
        self.curriculum_steps = curriculum_steps
        self.records: list[dict[str, Any]] = []

    def __call__(
        self, completions: list[Any], task_seed: list[int], **kwargs: Any
    ) -> list[float]:
        curriculum_weight = self._curriculum_weight(kwargs.get("trainer_state"))
        if len(completions) != len(task_seed):
            raise ValueError("completions and task_seed must have equal lengths")
        rewards: list[float] = []
        for completion, seed in zip(completions, task_seed, strict=True):
            completion_text = _completion_text(completion)
            try:
                action = parse_click_test_action(completion_text).render()
                valid_action = True
                parse_error = None
            except InvalidActionError as error:
                action = None
                valid_action = False
                parse_error = str(error)

            if not valid_action and not self.format_curriculum:
                reward = 0.0
                rewards.append(reward)
                self._record(
                    completion_text,
                    int(seed),
                    reward,
                    False,
                    False,
                    parse_error,
                    curriculum_weight,
                )
                continue

            environment = self.environment_factory()
            try:
                initial = environment.reset(seed=int(seed))
                target_action = _target_action(initial.observation.accessibility_tree)
                if valid_action:
                    result = environment.step(action)
                    success = result.done and result.reward > 0
                    reward = 1.0 if success else 0.25 * curriculum_weight
                else:
                    success = False
                    similarity = SequenceMatcher(
                        None, completion_text.strip(), target_action
                    ).ratio()
                    reward = round(0.2 * curriculum_weight * similarity, 6)
            finally:
                environment.close()
            rewards.append(reward)
            self._record(
                completion_text,
                int(seed),
                reward,
                valid_action,
                success,
                parse_error,
                curriculum_weight,
            )
        return rewards

    def _curriculum_weight(self, trainer_state: Any) -> float:
        if not self.format_curriculum:
            return 0.0
        if self.curriculum_steps is None:
            return 1.0
        step = int(getattr(trainer_state, "global_step", 0))
        return max(0.0, 1.0 - step / self.curriculum_steps)

    def _record(
        self,
        completion: str,
        seed: int,
        reward: float,
        valid_action: bool,
        success: bool,
        parse_error: str | None,
        curriculum_weight: float,
    ) -> None:
        self.records.append(
            {
                "completion": completion,
                "task_seed": seed,
                "reward": reward,
                "valid_action": valid_action,
                "success": success,
                "parse_error": parse_error,
                "curriculum_weight": curriculum_weight,
            }
        )


def make_click_test_reward(
    environment_factory: EnvironmentFactory,
    *,
    format_curriculum: bool = False,
    curriculum_steps: int | None = None,
) -> ClickTestReward:
    """Return a strict or explicitly shaped BrowserGym reward callable."""

    return ClickTestReward(
        environment_factory,
        format_curriculum=format_curriculum,
        curriculum_steps=curriculum_steps,
    )
