"""Transformers inference policy loaded only in model-enabled runtimes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from browser_control.evaluation import PolicyOutput
from browser_control.hardware import CudaDevice, inspect_cuda_device

DEFAULT_MODEL_ID = "LiquidAI/LFM2-350M"


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Named generation settings shared by evaluation entrypoints."""

    name: str
    do_sample: bool
    max_new_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not self.do_sample and any(
            value is not None for value in (self.temperature, self.top_p, self.min_p)
        ):
            raise ValueError("greedy decoding cannot set sampling parameters")
        if self.top_p is not None and self.min_p is not None:
            raise ValueError("choose top_p or min_p, not both")

    def generation_kwargs(self) -> dict[str, Any]:
        """Return only arguments meaningful for this decoding strategy."""

        kwargs: dict[str, Any] = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }
        for name in ("temperature", "top_p", "min_p", "repetition_penalty"):
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        return kwargs

    def grpo_kwargs(self) -> dict[str, Any]:
        """Translate a sampling decoder into TRL's GRPOConfig field names."""

        if not self.do_sample:
            raise ValueError("GRPO generation requires a sampling decoder")
        kwargs = self.generation_kwargs()
        del kwargs["do_sample"]
        kwargs["max_completion_length"] = kwargs.pop("max_new_tokens")
        return kwargs


TRAINING_DECODER = DecoderConfig(
    name="training",
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=32,
)
LEGACY_EVALUATION_DECODER = DecoderConfig(
    name="current-evaluation",
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=16,
)


@dataclass(slots=True)
class TransformersPolicy:
    model: Any
    tokenizer: Any
    torch: Any
    device: CudaDevice
    decoder: DecoderConfig = TRAINING_DECODER

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL_ID) -> TransformersPolicy:
        """Load an FP16/BF16 model onto the first validated CUDA device."""

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "The model runtime requires Colab's PyTorch and the `model` extra."
            ) from error

        device = inspect_cuda_device(torch)
        dtype = getattr(torch, device.training_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model.to(f"cuda:{device.index}")
        model.eval()
        return cls(model=model, tokenizer=tokenizer, torch=torch, device=device)

    def generate(
        self, messages: list[dict[str, str]], *, seed: int
    ) -> PolicyOutput:
        """Sample one short action using Liquid AI's recommended base settings."""

        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {
            name: tensor.to(f"cuda:{self.device.index}")
            for name, tensor in encoded.items()
        }
        prompt_length = int(encoded["input_ids"].shape[-1])
        started = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                **self.decoder.generation_kwargs(),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generation_seconds = time.monotonic() - started
        completion_ids = generated[0, prompt_length:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return PolicyOutput(
            text=text,
            generation_seconds=generation_seconds,
            prompt_tokens=prompt_length,
            completion_tokens=int(completion_ids.shape[-1]),
        )
