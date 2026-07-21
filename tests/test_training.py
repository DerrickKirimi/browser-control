from __future__ import annotations

import pytest

from browser_control.training import (
    LFM2_LORA_TARGET_MODULES,
    trainable_parameter_names,
    validate_lora_targets,
)


def test_expected_lfm2_targets_are_not_llama_mlp_names() -> None:
    assert LFM2_LORA_TARGET_MODULES == (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "w1",
        "w2",
        "w3",
    )


def test_lora_target_validation_counts_every_suffix() -> None:
    module_names = [
        f"model.layers.0.self_attn.{name}" for name in LFM2_LORA_TARGET_MODULES[:4]
    ] + [f"model.layers.0.feed_forward.{name}" for name in LFM2_LORA_TARGET_MODULES[4:]]

    counts = validate_lora_targets(module_names)

    assert counts == {name: 1 for name in LFM2_LORA_TARGET_MODULES}


def test_lora_target_validation_rejects_partial_attachment() -> None:
    with pytest.raises(ValueError, match="out_proj, w1, w2, w3"):
        validate_lora_targets(("layer.q_proj", "layer.k_proj", "layer.v_proj"))


class _Parameter:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


def test_trainable_parameters_must_be_adapter_only() -> None:
    names = trainable_parameter_names(
        (
            ("base.weight", _Parameter(False)),
            ("layer.q_proj.lora_A.default.weight", _Parameter(True)),
            ("layer.q_proj.lora_B.default.weight", _Parameter(True)),
        )
    )
    assert len(names) == 2

    with pytest.raises(RuntimeError, match="non-LoRA"):
        trainable_parameter_names((("base.weight", _Parameter(True)),))
    with pytest.raises(RuntimeError, match="no trainable"):
        trainable_parameter_names((("base.weight", _Parameter(False)),))
