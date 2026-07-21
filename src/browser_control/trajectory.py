"""Validated one-step trajectories and a transparent GRPO reference objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class OneStepTrajectory:
    """One sampled completion for one task instance.

    Prompt tokens are conditioning context. Completion tokens are policy-generated;
    only positions whose trainable mask is true contribute to the policy objective.
    Invalid action text remains a completion and receives reward zero.
    """

    task_id: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    trainable_mask: tuple[bool, ...]
    reward: float
    valid_action: bool
    terminal: bool
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if not self.prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if not self.completion_ids:
            raise ValueError("completion_ids must not be empty")
        expected = len(self.completion_ids)
        if len(self.behavior_logprobs) != expected:
            raise ValueError("behavior_logprobs must align with completion_ids")
        if len(self.trainable_mask) != expected:
            raise ValueError("trainable_mask must align with completion_ids")
        if not any(self.trainable_mask):
            raise ValueError("trajectory must contain at least one trainable token")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if any(not math.isfinite(value) for value in self.behavior_logprobs):
            raise ValueError("behavior_logprobs must be finite")
        if self.terminal and self.truncated:
            raise ValueError("a trajectory cannot be both terminal and truncated")
        if not self.valid_action and self.reward != 0.0:
            raise ValueError("invalid actions must receive reward zero")


@dataclass(frozen=True, slots=True)
class TrajectoryGroup:
    """Completions sampled for the same exact task prompt."""

    members: tuple[OneStepTrajectory, ...]

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("a GRPO group requires at least two members")
        first = self.members[0]
        if any(member.task_id != first.task_id for member in self.members[1:]):
            raise ValueError("all group members must share task_id")
        if any(member.prompt_ids != first.prompt_ids for member in self.members[1:]):
            raise ValueError("all group members must share prompt_ids")

    def advantages(self, *, epsilon: float = 1e-4) -> tuple[float, ...]:
        """Match TRL 0.29 group scaling: sample std plus epsilon."""

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        rewards = tuple(member.reward for member in self.members)
        reward_mean = fmean(rewards)
        variance = sum((reward - reward_mean) ** 2 for reward in rewards) / (
            len(rewards) - 1
        )
        if variance == 0.0:
            return tuple(0.0 for _ in rewards)
        scale = math.sqrt(variance) + epsilon
        return tuple((reward - reward_mean) / scale for reward in rewards)


def clipped_policy_loss(
    trajectory: OneStepTrajectory,
    current_logprobs: tuple[float, ...],
    advantage: float,
    *,
    clip_epsilon: float = 0.2,
) -> float:
    """Compute the masked, token-mean clipped surrogate loss for one completion."""

    if len(current_logprobs) != len(trajectory.completion_ids):
        raise ValueError("current_logprobs must align with completion_ids")
    if any(not math.isfinite(value) for value in current_logprobs):
        raise ValueError("current_logprobs must be finite")
    if not math.isfinite(advantage):
        raise ValueError("advantage must be finite")
    if clip_epsilon < 0:
        raise ValueError("clip_epsilon must be non-negative")

    objectives: list[float] = []
    lower = 1.0 - clip_epsilon
    upper = 1.0 + clip_epsilon
    for current, behavior, trainable in zip(
        current_logprobs,
        trajectory.behavior_logprobs,
        trajectory.trainable_mask,
        strict=True,
    ):
        if not trainable:
            continue
        ratio = math.exp(current - behavior)
        clipped_ratio = min(max(ratio, lower), upper)
        objectives.append(min(ratio * advantage, clipped_ratio * advantage))
    return -fmean(objectives)
