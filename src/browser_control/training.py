"""Explicit LoRA and GRPO settings for the two-update smoke run."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

LFM2_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "w1",
    "w2",
    "w3",
)


@dataclass(frozen=True, slots=True)
class SmokeTrainingConfig:
    model_id: str = "LiquidAI/LFM2-350M"
    seed: int = 23
    max_steps: int = 2
    num_generations: int = 4
    max_completion_length: int = 32
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 16


@dataclass(frozen=True, slots=True)
class ControlledTrainingConfig:
    model_id: str = "LiquidAI/LFM2-350M"
    seed: int = 23
    max_steps: int = 60
    curriculum_steps: int = 40
    num_generations: int = 4
    max_completion_length: int = 32
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 16

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0 < self.curriculum_steps < self.max_steps:
            raise ValueError("curriculum_steps must leave a strict-reward phase")
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least two")


def validate_lora_targets(
    module_names: Iterable[str],
    targets: tuple[str, ...] = LFM2_LORA_TARGET_MODULES,
) -> dict[str, int]:
    """Require every requested target suffix to occur in the loaded model."""

    names = tuple(module_names)
    counts = {
        target: sum(name == target or name.endswith(f".{target}") for name in names)
        for target in targets
    }
    missing = tuple(target for target, count in counts.items() if count == 0)
    if missing:
        raise ValueError(f"LoRA target modules absent from model: {', '.join(missing)}")
    return counts


def create_lora_config(
    config: SmokeTrainingConfig | ControlledTrainingConfig,
) -> Any:
    """Create the pinned PEFT configuration only in training-enabled runtimes."""

    try:
        from peft import LoraConfig
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the project training extra to create LoRA") from error
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LFM2_LORA_TARGET_MODULES),
    )


def trainable_parameter_names(named_parameters: Iterable[tuple[str, Any]]) -> tuple[str, ...]:
    """Fail unless at least one parameter is trainable and every one is LoRA-owned."""

    names = tuple(name for name, parameter in named_parameters if parameter.requires_grad)
    if not names:
        raise RuntimeError("LoRA attachment produced no trainable parameters")
    unexpected = tuple(name for name in names if "lora_" not in name)
    if unexpected:
        raise RuntimeError(
            "non-LoRA parameters are trainable: " + ", ".join(unexpected)
        )
    return names


def configure_model_for_training(model: Any) -> None:
    """Apply model-level settings that are not stable GRPOConfig arguments."""

    if not hasattr(model, "config"):
        raise TypeError("model must expose a config object")
    model.config.use_cache = False
