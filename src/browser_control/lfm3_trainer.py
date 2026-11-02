"""Lazy TRL trainer construction with explicit FP16 loss-scaler initialization."""

from __future__ import annotations


def low_scale_grpo_trainer_class():
    """Return the pinned trainer with scale 1 passed before Accelerate prepares it.

    The CPU fixture can check handler wiring, but only a real FP16 GPU update
    validates numerical stability. Do not replace the scaler after optimizer
    preparation: Accelerate's optimizer must share the original scaler object.
    """
    import trl
    from accelerate.utils import GradScalerKwargs

    if trl.__version__ != "1.16.0":
        raise ValueError(f"Expected TRL 1.16.0, got {trl.__version__}")

    class LowScaleGRPOTrainer(trl.GRPOTrainer):
        def _build_accelerator_args(self, **kwargs):
            result = super()._build_accelerator_args(**kwargs)
            handlers = list(result.get("kwargs_handlers", []))
            if any(isinstance(handler, GradScalerKwargs) for handler in handlers):
                raise ValueError("Loss scaler configuration must have one owner")
            result["kwargs_handlers"] = [*handlers, GradScalerKwargs(init_scale=1.0)]
            return result

    return LowScaleGRPOTrainer
