"""Pinned vision model identity and screenshot-only native coordinate prompts."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from browser_control.hardware import inspect_cuda_device
from browser_control.lfm3 import PromptTooLongError
from browser_control.transformers_policy import DecoderConfig
from browser_control.vision_browser import VisionObservation

MODEL_ID = "LiquidAI/LFM3-VL-2B"
MODEL_REVISION = "ba44bd51db20a2753f28c273a1571ccaa210d954"
COORDINATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click a screenshot location using normalized coordinates0–1000.",
            "parameters": {
                "type": "object",
                "properties": {
                    axis: {"type": "integer", "minimum": 0, "maximum": 1000}
                    for axis in ("x", "y")
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    }
]


def build_vision_messages(observation: VisionObservation):
    from PIL import Image

    image = Image.open(io.BytesIO(observation.screenshot_png))
    image.load()
    if image.mode != "RGB" or image.size != (observation.width, observation.height):
        raise ValueError("Screenshot dimensions/mode differ from the observation")
    text = observation.goal
    if observation.last_action_error:
        text += "\nLast action error: " + observation.last_action_error
    return [
        {
            "role": "system",
            "content": "Complete the browser task from the screenshot. Call exactly one provided tool. Coordinates x/y are integers0–1000, spanning the entire screenshot from left/top to right/bottom. Return only the tool call, without reasoning or prose.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        },
    ]


@dataclass(frozen=True, slots=True)
class VisionGeneration:
    text: str
    generation_seconds: float
    prompt_tokens: int
    completion_tokens: int
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    image_grid_thw: tuple[tuple[int, ...], ...]
    generation_logits_checks: int


@dataclass(slots=True)
class LFM3VisionPolicy:
    """Frozen bounded inference, with explicit decoder and raw token evidence."""

    model: Any
    processor: Any
    torch: Any
    decoder: DecoderConfig
    device: str = "cuda:0"
    max_prompt_tokens: int = 2048

    def __post_init__(self):
        from transformers import GenerationConfig

        if not 1 <= self.max_prompt_tokens <= 2048 or self.decoder.max_new_tokens > 128:
            raise ValueError("Vision inference caps prompt/completion at2048/128")
        self.model.requires_grad_(False).eval()
        # Match the explicit evaluation decoder, without inheriting checkpoint
        # temperature .1, top_k50 or repetition penalty1.1.
        tokenizer = self.processor.tokenizer
        self.model.generation_config = GenerationConfig(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_k=0,
            repetition_penalty=1.0,
        )

    @classmethod
    def load(cls, decoder: DecoderConfig):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if version("transformers") != "5.21.0":
            raise ValueError("Frozen vision runtime requires Transformers5.21.0")
        if (
            version("torch").split("+")[0] != "2.14.1"
            or version("torchvision").split("+")[0] != "0.29.1"
        ):
            raise ValueError(
                "Frozen vision runtime requires Torch2.14.1/Torchvision0.29.1"
            )
        device = inspect_cuda_device(torch)
        if "T4" not in device.name or device.training_dtype != "float16":
            raise ValueError("This bounded vision gate requires a T4 with FP16")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            dtype=torch.float16,
            attn_implementation="sdpa",
        ).to("cuda:0")
        return cls(model, processor, torch, decoder)

    def encode(self, observation: VisionObservation):
        encoded = self.processor.apply_chat_template(
            build_vision_messages(observation),
            tools=COORDINATE_TOOLS,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        length = int(encoded["input_ids"].shape[-1])
        if length > self.max_prompt_tokens:
            raise PromptTooLongError(length, self.max_prompt_tokens)
        if "pixel_values" not in encoded or "image_grid_thw" not in encoded:
            raise ValueError("Vision processor omitted image inputs")
        if not self.torch.isfinite(encoded["pixel_values"]).all():
            raise ValueError("Nonfinite processed image")
        return {
            key: value.to(device=self.device, dtype=self.model.dtype)
            if key == "pixel_values"
            else value.to(self.device)
            for key, value in encoded.items()
        }

    def check_logits(self, observation: VisionObservation):
        encoded = self.encode(observation)
        with self.torch.inference_mode():
            logits = self.model(**encoded, use_cache=False, logits_to_keep=1).logits
        if not self.torch.isfinite(logits).all():
            raise RuntimeError(
                "Nonfinite vision logits; stop before collecting results"
            )
        return {
            "finite": True,
            "max_abs_logit": float(logits.abs().max()),
            "prompt_tokens": int(encoded["input_ids"].shape[-1]),
            "image_tokens": int(
                (encoded["input_ids"] == self.model.config.image_token_id).sum()
            ),
        }

    def generate(self, observation: VisionObservation, *, seed: int):
        self.torch.manual_seed(seed)
        cuda = self.device.startswith("cuda")
        if cuda:
            self.torch.cuda.manual_seed_all(seed)
        encoded = self.encode(observation)
        length = int(encoded["input_ids"].shape[-1])
        if cuda:
            self.torch.cuda.synchronize()
        started = time.monotonic()
        checks = 0

        def check_forward(_module, _arguments, output):
            nonlocal checks
            if not self.torch.isfinite(output.logits[:, -1, :]).all():
                raise RuntimeError("Nonfinite vision generation logits")
            checks += 1

        hook = self.model.register_forward_hook(check_forward)
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded, **self.decoder.generation_kwargs()
                )
        finally:
            hook.remove()
        if cuda:
            self.torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        completion = generated[0, length:]
        if checks != int(completion.shape[-1]):
            raise RuntimeError(
                "Vision generation did not check every next-token forward"
            )
        return VisionGeneration(
            text=self.processor.tokenizer.decode(completion, skip_special_tokens=True),
            generation_seconds=elapsed,
            prompt_tokens=length,
            completion_tokens=int(completion.shape[-1]),
            prompt_token_ids=tuple(encoded["input_ids"][0].tolist()),
            completion_token_ids=tuple(completion.tolist()),
            image_grid_thw=tuple(
                tuple(row) for row in encoded["image_grid_thw"].tolist()
            ),
            generation_logits_checks=checks,
        )
