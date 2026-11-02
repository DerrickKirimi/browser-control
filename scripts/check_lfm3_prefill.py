"""Validate chunked eager prefill on development prompts and a 16K stress input."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from browser_control.book_flight import BOOK_FLIGHT_TOOLS
from browser_control.lfm3 import LFM3Policy
from browser_control.transformers_policy import DecoderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = LFM3Policy.load(DecoderConfig("greedy", False, 2), "native-tools")
    policy.tools = BOOK_FLIGHT_TOOLS
    torch = policy.torch
    records = [
        r
        for r in json.loads(args.records.read_text())
        if r["task"] == "book-flight-nodelay"
    ]
    short = records[0]["steps"][0]["messages"]
    longest = records[1]["steps"][-1]["messages"]
    report = {"prefill_chunk_size": 512, "attention": "eager", "comparisons": []}

    def chunked(messages):
        encoded = policy.encode(messages)
        with torch.inference_mode():
            generated = policy.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=2,
                prefill_chunk_size=512,
                return_dict_in_generate=True,
                output_scores=True,
            )
        assert all(torch.isfinite(score).all() for score in generated.scores)
        return generated.scores[0].float(), int(encoded["input_ids"].shape[-1])

    for messages in (short, longest):
        with torch.inference_mode():
            full = (
                policy.model(**policy.encode(messages), logits_to_keep=1)
                .logits[:, -1, :]
                .float()
            )
        chunk, length = chunked(messages)
        difference = (full - chunk).abs()
        entry = {
            "prompt_tokens": length,
            "top1_agrees": bool(full.argmax(-1).item() == chunk.argmax(-1).item()),
            "max_absolute_difference": difference.max().item(),
            "mean_absolute_difference": difference.mean().item(),
        }
        print(entry, flush=True)
        report["comparisons"].append(entry)
        assert entry["top1_agrees"] and entry["max_absolute_difference"] < 0.5, entry
        del full, chunk, difference
        torch.cuda.empty_cache()
    tree = records[1]["steps"][5]["observation"]["accessibility_tree"]
    stress = None
    for repeats in range(1, 30):
        candidate = copy.deepcopy(longest) + [
            {
                "role": "user",
                "content": "Synthetic memory stress context:\n"
                + (tree + "\n") * repeats,
            }
        ]
        if policy.encode(candidate)["input_ids"].shape[-1] > 16384:
            break
        stress = candidate
    assert stress is not None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    _, length = chunked(stress)
    assert 12000 < length <= 16384, length
    report.update(
        status="passed",
        stress_prompt_tokens=length,
        stress_peak_vram_bytes=torch.cuda.max_memory_allocated(),
        stress_seconds=time.monotonic() - started,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
