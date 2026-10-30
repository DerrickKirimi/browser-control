"""Gate LoRA training on finite, nonzero T4 updates at realistic context lengths."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from preflight_lfm3_attention import LONG_PROMPT_TOKENS, LORA, load, lora_attempt

from browser_control.book_flight import BOOK_FLIGHT_TOOLS, BookFlightAction
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    from peft import LoraConfig, get_peft_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the existing report; use a new output path")
    records = [
        r
        for r in json.loads(args.records.read_text())
        if r["task"] == "book-flight-nodelay"
    ]
    short = records[0]["steps"][0]["messages"]
    longest = max(
        (step["messages"] for record in records for step in record["steps"]),
        key=lambda messages: len(json.dumps(messages)),
    )
    policy = load("sdpa")
    torch = policy.torch
    torch.manual_seed(23)
    model = get_peft_model(policy.model, LoraConfig(**LORA))
    model.config.use_cache = False
    model.train()
    tree = records[1]["steps"][5]["observation"]["accessibility_tree"]
    stress = list(longest)
    for repeats in range(1, 30):
        stress = [
            *longest,
            {
                "role": "user",
                "content": "Synthetic memory stress context:\n"
                + (tree + "\n") * repeats,
            },
        ]
        if policy.encode(stress)["input_ids"].shape[-1] >= LONG_PROMPT_TOKENS:
            break
    report = {
        "status": "running",
        "update_passed": False,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "attention": "sdpa",
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "trl")
        },
        "device": torch.cuda.get_device_name(),
        "lora": LORA,
        "trainable_tensors": sum(p.requires_grad for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "attempts": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(
            json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n"
        )

    save()
    action = BookFlightAction("click", {"bid": "392"}).message()
    # Short no-checkpoint control isolates loss scaling from checkpointing.
    # Each attempt restores the identical initial adapters, including on errors.
    for name, prompt, checkpointing in (
        ("short-control", short, False),
        ("observed-long", longest, True),
        ("synthetic-12k", stress, True),
    ):
        prefix = policy.encode(prompt)["input_ids"]
        full = policy.tokenizer.apply_chat_template(
            [*prompt, action],
            tools=BOOK_FLIGHT_TOOLS,
            enable_thinking=False,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"].to("cuda:0")
        if not torch.equal(full[:, : prefix.shape[-1]], prefix):
            raise ValueError("Chat template prefix changed; target mask is invalid")
        target_length = full.shape[-1] - prefix.shape[-1]
        if target_length <= 0 or full[0, -1].item() != policy.tokenizer.eos_token_id:
            raise ValueError("Expected a nonempty canonical tool call ending in EOS")
        for scale in (65536.0, 128.0, 1.0):
            row = lora_attempt(
                policy, model, full, target_length, checkpointing, initial_scale=scale
            )
            row.update(case=name, target_tokens=target_length, prefix_matches=True)
            report["attempts"].append(row)
            save()
            if row["update_passed"]:
                break
    report["update_passed"] = all(
        any(row["case"] == name and row["update_passed"] for row in report["attempts"])
        for name in ("short-control", "observed-long", "synthetic-12k")
    )
    report["status"] = "passed" if report["update_passed"] else "failed"
    save()
    if not report["update_passed"]:
        raise SystemExit("Finite-update gate failed; training remains blocked")


if __name__ == "__main__":
    main()
