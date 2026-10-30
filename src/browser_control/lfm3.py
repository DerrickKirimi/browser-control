"""LFM3 click-test interfaces; no model imports required for validation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from browser_control.actions import (
    InvalidActionError,
    ParsedAction,
    parse_click_test_action,
)
from browser_control.environment import BrowserObservation
from browser_control.evaluation import PolicyOutput
from browser_control.hardware import inspect_cuda_device
from browser_control.prompts import build_click_test_messages
from browser_control.transformers_policy import DecoderConfig

MODEL_ID = "LiquidAI/LFM3-1.7B"
MODEL_REVISION = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element by its browser element ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bid": {
                        "type": "string",
                        "description": "ID from the accessibility tree",
                    }
                },
                "required": ["bid"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Wait without interacting.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def build_lfm3_messages(observation: BrowserObservation) -> list[dict[str, str]]:
    """Native interface has no concrete answer example or tree truncation."""
    return [
        {
            "role": "system",
            "content": (
                "Complete the browser task by calling exactly one provided tool. "
                "Use browser element IDs from the accessibility tree. Return only the tool call."
            ),
        },
        build_click_test_messages(observation)[1],
    ]


def parse_lfm3_action(output: str) -> ParsedAction:
    """Accept one documented XML call, then apply the existing AST validator.

    This restricted executor rejects prose, reasoning, unknown parameters and
    multiple calls. It never executes generated Python or repairs invalid output.
    """
    match = re.fullmatch(
        r"\s*<tool_call>\s*<function=(click|noop)>\s*(.*?)\s*</function>\s*</tool_call>\s*",
        output,
        flags=re.DOTALL,
    )
    if match is None:
        raise InvalidActionError("expected exactly one click/noop XML tool call")
    name, body = match.groups()
    if name == "noop":
        if body.strip():
            raise InvalidActionError("noop takes no parameters")
        return parse_click_test_action("noop()")
    parameter = re.fullmatch(
        r"<parameter=bid>\s*([^<>]*?)\s*</parameter>", body.strip()
    )
    if parameter is None:
        raise InvalidActionError("click requires exactly one bid parameter")
    return parse_click_test_action(f"click({parameter.group(1).strip()!r})")


class PromptTooLongError(ValueError):
    def __init__(self, tokens: int, limit: int):
        self.tokens, self.limit = tokens, limit
        super().__init__(
            f"prompt has {tokens} tokens; budget is {limit}; no truncation applied"
        )


@dataclass
class LFM3Policy:
    model: Any
    tokenizer: Any
    torch: Any
    decoder: DecoderConfig
    interface: str
    revision: str = MODEL_REVISION
    tools: list[dict[str, Any]] | None = None
    max_prompt_tokens: int | None = None
    enable_thinking: bool = False

    @classmethod
    def load(
        cls,
        decoder: DecoderConfig,
        interface: str,
        attn_implementation: str = "eager",
    ) -> LFM3Policy:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if interface not in {"strict-text", "native-tools"}:
            raise ValueError("unknown action interface")
        device = inspect_cuda_device(torch)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            padding_side="left",
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=True,
                dtype=getattr(torch, device.training_dtype),
                attn_implementation=attn_implementation,
            )
            .to("cuda:0")
            .eval()
        )
        # Do not inherit the checkpoint's penalty=1.1, temperature=.1, top_k=50.
        model.generation_config = GenerationConfig(
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            top_k=0,
            repetition_penalty=1.0,
        )
        return cls(model, tokenizer, torch, decoder, interface)

    def encode(self, messages: list[dict[str, Any]]) -> Any:
        kwargs = (
            {"tools": self.tools if self.tools is not None else TOOLS}
            if self.interface == "native-tools"
            else {}
        )
        encoded = self.tokenizer.apply_chat_template(
            messages,
            **kwargs,
            enable_thinking=self.enable_thinking,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        length = encoded["input_ids"].shape[-1]
        if self.max_prompt_tokens is not None and length > self.max_prompt_tokens:
            raise PromptTooLongError(int(length), self.max_prompt_tokens)
        return encoded.to("cuda:0")

    def check_logits(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Fail early on FP16 overflow before collecting success statistics."""
        encoded = self.encode(messages)
        with self.torch.inference_mode():
            logits = self.model(**encoded, logits_to_keep=1).logits[:, -1, :].float()
        finite = bool(self.torch.isfinite(logits).all().item())
        if not finite:
            raise RuntimeError(
                "non-finite LFM3 logits; stop before evaluation/training"
            )
        return {"finite": finite, "max_abs_logit": logits.abs().max().item()}

    def generate(self, messages: list[dict[str, Any]], *, seed: int) -> PolicyOutput:
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        encoded = self.encode(messages)
        length = encoded["input_ids"].shape[-1]
        self.torch.cuda.synchronize()
        started = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded, **self.decoder.generation_kwargs()
            )
        self.torch.cuda.synchronize()
        completion = generated[0, length:]
        return PolicyOutput(
            text=self.tokenizer.decode(completion, skip_special_tokens=True),
            generation_seconds=time.monotonic() - started,
            prompt_tokens=int(length),
            completion_tokens=int(completion.shape[-1]),
        )


def frozen_gate(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    """Require the complete preregistered matrix before making a decision."""
    expected = {
        (interface, mode, seed)
        for interface in ("strict-text", "native-tools")
        for mode in ("greedy", "sampling")
        for seed in (23, 47, 71)
    }
    keys = [
        (cell["interface"], cell["decoder"]["name"], cell["seed"])
        for cell in conditions
    ]
    complete = len(keys) == len(expected) and set(keys) == expected
    eligible = complete and all(
        cell["numerical_check"]["finite"]
        and len(cell["evaluation"]["episodes"])
        == (50 if cell["decoder"]["name"] == "sampling" else 1)
        for cell in conditions
    )
    passed = eligible and all(
        sum(episode["success"] for episode in cell["evaluation"]["episodes"])
        >= (48 if cell["decoder"]["name"] == "sampling" else 1)
        for cell in conditions
        if cell["interface"] == "native-tools"
    )
    return {
        "eligible": eligible,
        "passed": passed,
        "decision": (
            "proceed-to-book-flight"
            if passed
            else "inspect-format-length-and-numerics"
            if eligible
            else "incomplete-or-nonstandard-run"
        ),
    }
