"""Transformers inference policy loaded only in model-enabled runtimes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from browser_control.evaluation import PolicyOutput
from browser_control.hardware import CudaDevice, inspect_cuda_device

DEFAULT_MODEL_ID = "LiquidAI/LFM2-350M"


@dataclass(slots=True)
class TransformersPolicy:
    model: Any
    tokenizer: Any
    torch: Any
    device: CudaDevice

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
                do_sample=True,
                temperature=0.3,
                min_p=0.15,
                repetition_penalty=1.05,
                max_new_tokens=16,
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
