"""Measure full native LFM3 prompt lengths on scripted development traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from browser_control.book_flight import BOOK_FLIGHT_TOOLS
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    measurements = []
    for episode in records:
        lengths = [
            int(
                tokenizer.apply_chat_template(
                    step["messages"],
                    tools=BOOK_FLIGHT_TOOLS,
                    enable_thinking=False,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )["input_ids"].shape[-1]
            )
            for step in episode["steps"]
        ]
        assert all(length >= 100 for length in lengths), (
            "invalid token-length measurement"
        )
        measurements.append(
            {
                "task": episode["task"],
                "seed": episode["task_seed"],
                "prompt_tokens_by_step": lengths,
                "action_errors": sum(bool(s["error"]) for s in episode["steps"]),
            }
        )
    report = {
        "model_revision": MODEL_REVISION,
        "measurements": measurements,
        "max_prompt_tokens": max(max(m["prompt_tokens_by_step"]) for m in measurements),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["max_prompt_tokens"] <= 16384, (
        "Revise the context budget before model evaluation"
    )


if __name__ == "__main__":
    main()
