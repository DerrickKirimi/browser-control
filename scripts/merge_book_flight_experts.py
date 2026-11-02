"""Combine disjoint training-only expert collections, preserving source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from browser_control import book_flight_solver
from browser_control.book_flight_corpus import merge_expert_corpora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing corpus evidence")
    raw = [path.read_bytes() for path in args.corpora]
    merged = merge_expert_corpora(
        [json.loads(data) for data in raw],
        solver_sha256=hashlib.sha256(
            Path(book_flight_solver.__file__).read_bytes()
        ).hexdigest(),
    )
    merged["source_corpora"] = [
        {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in zip(args.corpora, raw, strict=True)
    ]
    merged["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    print(
        {
            "seeds": len(merged["seeds"]),
            "eligible_episodes": merged["eligible_episodes"],
            "examples": len(merged["examples"]),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
    )


if __name__ == "__main__":
    main()
