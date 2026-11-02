"""Check the pinned TRL 1.16 loss/mask contract on CPU, without model weights."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

from browser_control.trajectory import OneStepTrajectory, clipped_policy_loss


def main() -> None:
    import torch
    import trl
    from trl.trainer.grpo_trainer import nanstd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert trl.__version__ == "1.16.0", trl.__version__
    checks = []

    def loss(
        current,
        *,
        advantage,
        completion_mask=(1, 1, 0),
        tool_mask=None,
        training=False,
        accumulation=1,
    ):
        trainer = trl.GRPOTrainer.__new__(trl.GRPOTrainer)
        trainer.top_entropy_quantile = 1.0
        trainer.off_policy_mask_threshold = None
        trainer.importance_sampling_level = "token"
        trainer.beta = 0.0
        trainer.loss_type = "grpo"
        trainer.epsilon_low = trainer.epsilon_high = 0.2
        trainer.args = SimpleNamespace(delta=None)
        trainer.use_vllm = trainer.vllm_importance_sampling_correction = False
        trainer.aux_loss_enabled = trainer._entropy_bonus_enabled = False
        trainer.current_gradient_accumulation_steps = accumulation
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer.accelerator = SimpleNamespace(
            gather=lambda value: value, reduce=lambda value, **kwargs: value
        )
        model = SimpleNamespace(training=training)
        trainer.model = model
        values = torch.tensor([current], dtype=torch.float32, requires_grad=True)
        seen = {}

        def fixed_logprobs(self, model, ids, attention, keep, **kwargs):
            seen["ids"] = ids.tolist()
            seen["attention"] = attention.tolist()
            return values, torch.zeros_like(values), None

        trainer._get_per_token_logps_and_entropies = MethodType(fixed_logprobs, trainer)
        inputs = {
            "prompt_ids": torch.tensor([[10, 11]]),
            "prompt_mask": torch.tensor([[1, 1]]),
            "completion_ids": torch.tensor([[20, 21, 0]]),
            "completion_mask": torch.tensor([completion_mask]),
            "old_per_token_logps": torch.tensor([[-3.0, -3.0, -100.0]]),
            "advantages": torch.tensor([advantage]),
        }
        if tool_mask is not None:
            inputs["tool_mask"] = torch.tensor([tool_mask])
        result = trainer._compute_loss(model, inputs)
        result.backward()
        assert torch.isfinite(result)
        assert torch.isfinite(values.grad).all()
        return result.item(), values.grad[0].tolist(), seen

    trajectory = OneStepTrajectory(
        task_id="contract",
        prompt_ids=(10, 11),
        completion_ids=(20, 21, 0),
        behavior_logprobs=(-3.0, -3.0, -100.0),
        trainable_mask=(True, True, False),
        reward=0.0,
        valid_action=False,
        terminal=True,
    )
    current = (-3 + math.log(1.5), -3 + math.log(0.5), -80.0)
    for advantage in (2.0, -2.0):
        expected = clipped_policy_loss(trajectory, current, advantage)
        actual, gradient, _ = loss(current, advantage=advantage)
        assert math.isclose(actual, expected, abs_tol=1e-6), (actual, expected)
        assert gradient[2] == 0.0, gradient
        checks.append({"check": f"clipping-advantage-{advantage}", "loss": actual})

    actual, gradient, seen = loss(
        (-3 + math.log(1.1), -3 + math.log(0.9), -80.0),
        advantage=2.0,
        tool_mask=(1, 0, 1),
    )
    assert math.isclose(actual, -2.2, abs_tol=1e-6)
    assert gradient[0] != 0.0 and gradient[1:] == [0.0, 0.0], gradient
    # Observation token 21 remains context but contributes no policy gradient.
    assert seen["ids"] == [[10, 11, 20, 21, 0]]
    assert seen["attention"] == [[1, 1, 1, 1, 0]]
    checks.append({"check": "observation-and-padding-mask", "loss": actual})

    for name, kwargs in (
        ("fully-masked-truncation", {"completion_mask": (0, 0, 0)}),
        ("zero-advantage", {}),
    ):
        actual, gradient, _ = loss(
            current, advantage=0.0 if not kwargs else 2.0, **kwargs
        )
        assert actual == 0.0 and gradient == [0.0, 0.0, 0.0], (actual, gradient)
        checks.append({"check": name, "loss": actual})

    evaluation, _, _ = loss(current, advantage=2.0)
    training, _, _ = loss(current, advantage=2.0, training=True, accumulation=4)
    assert math.isclose(training, evaluation / 4, abs_tol=1e-6)
    checks.append({"check": "gradient-accumulation-scaling", "loss": training})

    rewards = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    standardized = (rewards - rewards.mean(dim=1, keepdim=True)) / (
        nanstd(rewards, dim=1).unsqueeze(1) + 1e-4
    )
    expected = (rewards[0] - 1 / 3) / (math.sqrt(1 / 3) + 1e-4)
    torch.testing.assert_close(standardized[0], expected)
    assert standardized[1].tolist() == [0.0, 0.0, 0.0]
    checks.append({"check": "sample-std-and-equal-reward-group"})

    source = Path(inspect.getsourcefile(trl.GRPOTrainer))
    report = {
        "status": "passed",
        "trl_version": trl.__version__,
        "torch_version": torch.__version__,
        "device": "cpu",
        "trainer_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "configuration": {
            "loss_type": "grpo",
            "scale_rewards": "group",
            "beta": 0.0,
            "importance_sampling_level": "token",
            "epsilon": 0.2,
        },
        "checks": checks,
        "scope": "isolated real loss method and normalization helper; no rollout or optimizer validation",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
