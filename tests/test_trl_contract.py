"""Contract checks against the pinned TRL implementation.

These tests require the optional ``training`` extra. The normal CPU suite skips
them, while the dedicated trainer-contract command must report them as passing.
"""

from __future__ import annotations

import math
from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

from browser_control.trajectory import (  # noqa: E402
    OneStepTrajectory,
    TrajectoryGroup,
    clipped_policy_loss,
)


def _trajectory(*, reward: float = 0.0) -> OneStepTrajectory:
    return OneStepTrajectory(
        task_id="click-test:23",
        prompt_ids=(10, 11),
        completion_ids=(20, 21, 0),
        behavior_logprobs=(0.0, 0.0, -100.0),
        trainable_mask=(True, True, False),
        reward=reward,
        valid_action=True,
        terminal=True,
    )


def _real_trl_loss(
    current: tuple[float, ...],
    old: tuple[float, ...],
    mask: tuple[float, ...],
    *,
    advantage: float,
) -> float:
    trainer = trl.GRPOTrainer.__new__(trl.GRPOTrainer)
    trainer.top_entropy_quantile = 1.0
    trainer.off_policy_mask_threshold = None
    trainer.importance_sampling_level = "token"
    trainer.beta = 0.0
    trainer.loss_type = "grpo"
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    trainer.args = SimpleNamespace(delta=None)
    trainer.use_vllm = False
    trainer.vllm_importance_sampling_correction = False
    trainer._metrics = {"eval": defaultdict(list)}
    trainer.accelerator = SimpleNamespace(gather=lambda value: value)
    model = SimpleNamespace(training=False)
    trainer.model = model

    def fixed_logprobs(self, *args, **kwargs):
        del self, args, kwargs
        values = torch.tensor([current], dtype=torch.float32)
        return values, torch.zeros_like(values)

    trainer._get_per_token_logps_and_entropies = MethodType(fixed_logprobs, trainer)
    inputs = {
        "prompt_ids": torch.tensor([[10, 11]]),
        "prompt_mask": torch.tensor([[1, 1]]),
        "completion_ids": torch.tensor([[20, 21, 0]]),
        "completion_mask": torch.tensor([mask]),
        "old_per_token_logps": torch.tensor([old]),
        "advantages": torch.tensor([advantage]),
    }
    return trainer._compute_loss(model, inputs).item()


def test_advantages_match_pinned_trl_torch_operations() -> None:
    group = TrajectoryGroup(
        (_trajectory(reward=0.0), _trajectory(reward=0.0), _trajectory(reward=1.0))
    )
    rewards = torch.tensor([0.0, 0.0, 1.0])
    trl_style = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    assert torch.tensor(group.advantages()) == pytest.approx(trl_style)


def test_real_trl_loss_matches_hand_computed_reference() -> None:
    assert trl.__version__ == "0.29.0"
    trajectory = _trajectory()
    current = (math.log(1.5), math.log(0.5), 100.0)
    expected = clipped_policy_loss(trajectory, current, advantage=2.0)

    actual = _real_trl_loss(
        current,
        (0.0, 0.0, -100.0),
        (1.0, 1.0, 0.0),
        advantage=2.0,
    )

    assert actual == pytest.approx(expected, abs=1e-6)


def test_real_trl_loss_excludes_fully_masked_truncated_completion() -> None:
    actual = _real_trl_loss(
        (100.0, 100.0, 100.0),
        (-100.0, -100.0, -100.0),
        (0.0, 0.0, 0.0),
        advantage=2.0,
    )

    assert actual == 0.0
