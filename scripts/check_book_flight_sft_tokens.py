"""Validate every expert label boundary with the pinned tokenizer on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from browser_control import book_flight_sft
from browser_control.book_flight_sft import encode_example, expert_examples
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior token-validation evidence")
    corpus = json.loads(args.corpus.read_text())
    if corpus["status"] != "complete":
        parser.error("Corpus collection is incomplete")
    examples = [example for r in corpus["records"] for example in expert_examples(r)]
    if not examples or examples != corpus["examples"]:
        parser.error("Expected nonempty clean expert examples from training seeds")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    encoded = [encode_example(tokenizer, example) for example in examples]
    for row in encoded:
        boundary = row["prompt_tokens"]
        if row["labels"][:boundary] != [-100] * boundary:
            raise ValueError("Prompt tokens received labels")
        if row["labels"][boundary:] != row["input_ids"][boundary:]:
            raise ValueError("Current-call labels do not match their token IDs")
        if row["labels"][-1] != tokenizer.eos_token_id:
            raise ValueError("Current-call labels must end at EOS")
    report = {
        "status": "passed",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "examples": len(encoded),
        "skills": dict(Counter(e["skill"] for e in examples)),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "encoder_sha256": hashlib.sha256(
            Path(book_flight_sft.__file__).read_bytes()
        ).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "tokenizers", "huggingface-hub")
        },
        "max_prompt_tokens": max(e["prompt_tokens"] for e in encoded),
        "max_target_tokens": max(e["target_tokens"] for e in encoded),
        "max_full_tokens": max(len(e["input_ids"]) for e in encoded),
    }
    if args.records is not None:
        from check_lfm3_gradients import gradient_cases

        from browser_control.book_flight import BookFlightAction

        records = [
            r
            for r in json.loads(args.records.read_text())
            if r["task"] == "book-flight-nodelay"
        ]
        action = BookFlightAction("click", {"bid": "392"}).message()
        report["gradient_cases"] = [
            {
                "case": name,
                "full_tokens": len(row["input_ids"]),
                "prompt_tokens": row["prompt_tokens"],
                "target_tokens": row["target_tokens"],
            }
            for name, prompt, _ in gradient_cases(tokenizer, records)
            for row in [
                encode_example(tokenizer, {"messages": prompt, "assistant": action})
            ]
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
