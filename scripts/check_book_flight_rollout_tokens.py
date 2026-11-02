"""Check external feedback suffixes against complete pinned chat templates on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_control.book_flight import BookFlightAction
from browser_control.book_flight_rollout import chat_ids, feedback_suffix
from browser_control.environment import BrowserObservation
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the previous token-boundary check")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    corpus = json.loads(args.corpus.read_text())
    checked = 0
    for record in corpus["records"]:
        for step, following in zip(record["steps"], record["steps"][1:]):
            call = step["assistant"]["tool_calls"][0]["function"]
            action = BookFlightAction(call["name"], call["arguments"])
            closed = chat_ids(
                tokenizer, [*step["messages"], step["assistant"]], generation=False
            )
            continued = chat_ids(tokenizer, following["messages"], generation=True)
            eos_end = len(closed) - closed[::-1].index(tokenizer.eos_token_id)
            actual = closed[:eos_end] + list(
                feedback_suffix(
                    tokenizer, action, BrowserObservation(**following["observation"])
                )
            )
            if actual != continued:
                raise ValueError(
                    f"Feedback suffix mismatch at seed {record['task_seed']}"
                )
            checked += 1
    report = {
        "status": "passed",
        "boundaries": checked,
        "model_revision": MODEL_REVISION,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
