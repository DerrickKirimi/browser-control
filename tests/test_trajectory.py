from __future__ import annotations

import math

import pytest

from browser_control.trajectory import (
    OneStepTrajectory,
    TrajectoryGroup,
    clipped_policy_loss,
)


def _trajectory(
    *,
    task_id: str = "click-test:23",
    prompt_ids: tuple[int, ...] = (10, 11),
    completion_ids: tuple[int, ...] = (20, 21, 0),
    behavior_logprobs: tuple[float, ...] = (-1.0, -2.0, 0.0),
    trainable_mask: tuple[bool, ...] = (True, True, False),
    reward: float = 0.0,
    valid_action: bool = True,
    terminal: bool = True,
    truncated: bool = False,
) -> OneStepTrajectory:
    return OneStepTrajectory(
        task_id=task_id,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        behavior_logprobs=behavior_logprobs,
        trainable_mask=trainable_mask,
        reward=reward,
        valid_action=valid_action,
        terminal=terminal,
        truncated=truncated,
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"behavior_logprobs": (-1.0,)}, "behavior_logprobs"),
        ({"trainable_mask": (True,)}, "trainable_mask"),
        ({"trainable_mask": (False, False, False)}, "trainable token"),
        ({"reward": math.inf}, "reward must be finite"),
        ({"behavior_logprobs": (-1.0, math.nan, 0.0)}, "must be finite"),
        ({"terminal": True, "truncated": True}, "both terminal and truncated"),
    ],
)
def test_trajectory_rejects_misalignment(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _trajectory(**override)  # type: ignore[arg-type]


def test_invalid_completion_is_retained_but_cannot_receive_reward() -> None:
    invalid = _trajectory(valid_action=False, reward=0.0, terminal=False)
    assert invalid.completion_ids == (20, 21, 0)
    assert invalid.trainable_mask == (True, True, False)

    with pytest.raises(ValueError, match="invalid actions must receive reward zero"):
        _trajectory(valid_action=False, reward=1.0)


def test_group_requires_same_task_and_exact_prompt() -> None:
    first = _trajectory()
    with pytest.raises(ValueError, match="task_id"):
        TrajectoryGroup((first, _trajectory(task_id="click-test:24")))
    with pytest.raises(ValueError, match="prompt_ids"):
        TrajectoryGroup((first, _trajectory(prompt_ids=(10, 12))))


def test_group_advantages_match_trl_sample_standard_deviation() -> None:
    group = TrajectoryGroup(
        (_trajectory(reward=0.0), _trajectory(reward=0.0), _trajectory(reward=1.0))
    )
    advantages = group.advantages(epsilon=1e-12)
    assert advantages == pytest.approx(
        (-1 / math.sqrt(3), -1 / math.sqrt(3), 2 / math.sqrt(3))
    )
    assert sum(advantages) == pytest.approx(0.0)


def test_equal_reward_group_has_zero_advantages() -> None:
    group = TrajectoryGroup((_trajectory(reward=1.0), _trajectory(reward=1.0)))
    assert group.advantages() == (0.0, 0.0)


def test_hand_computed_clipped_loss_masks_padding() -> None:
    trajectory = _trajectory(
        behavior_logprobs=(0.0, 0.0, -100.0),
        trainable_mask=(True, True, False),
    )
    current = (math.log(1.5), math.log(0.5), 100.0)

    # For positive A=2: min(1.5*2, 1.2*2)=2.4 and min(0.5*2, 0.8*2)=1.0.
    assert clipped_policy_loss(trajectory, current, 2.0) == pytest.approx(-1.7)
    # For negative A=-2, clipping selects the more pessimistic objective per token.
    assert clipped_policy_loss(trajectory, current, -2.0) == pytest.approx(2.3)


def test_loss_requires_logprob_alignment_and_finite_values() -> None:
    trajectory = _trajectory()
    with pytest.raises(ValueError, match="align"):
        clipped_policy_loss(trajectory, (-1.0,), 1.0)
    with pytest.raises(ValueError, match="finite"):
        clipped_policy_loss(trajectory, (-1.0, math.nan, 0.0), 1.0)
