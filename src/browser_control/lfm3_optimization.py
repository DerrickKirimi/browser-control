"""Finite LoRA optimizer steps, including legitimate finite zero gradients."""

from __future__ import annotations

from typing import Any


class InvalidLoRAGradients(RuntimeError):
    def __init__(self, names: list[str]):
        self.names = names
        super().__init__("Missing/nonfinite gradients; optimizer step refused")


def checked_lora_step(
    parameters: dict[str, Any], optimizer: Any, scaler: Any
) -> dict[str, float | bool]:
    """Keep ordinary AdamW momentum semantics; zero signal is reported separately.

    The caller validates the LoRA parameter set and requires changed weights over
    the run. A zero gradient is distinct from a missing or nonfinite gradient.
    """
    import torch

    scaler.unscale_(optimizer)
    bad = [
        name
        for name, parameter in parameters.items()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
    ]
    if bad:
        raise InvalidLoRAGradients(bad)
    norm = torch.nn.utils.clip_grad_norm_(
        parameters.values(), 1.0, error_if_nonfinite=True
    )
    # A float32 norm can underflow before its individual entries do.
    nonzero = norm.item() > 0 or any(p.grad.any().item() for p in parameters.values())
    # Even a zero gradient can produce a momentum update. Do not silently change
    # optimizer semantics or count this as a step with new gradient signal.
    scaler.step(optimizer)
    scaler.update()
    if any(not torch.isfinite(p).all().item() for p in parameters.values()):
        raise RuntimeError("Nonfinite adapter weights after update")
    return {
        "unclipped_gradient_norm": norm.item(),
        "nonzero_gradient": nonzero,
        "zero_gradient_norm": norm.item() == 0,
        "scale": scaler.get_scale(),
    }
