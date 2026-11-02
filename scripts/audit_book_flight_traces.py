"""Audit visible decisions in a saved native book-flight evaluation, offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_control.book_flight_trace_audit import audit_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous audit evidence")
    data = args.evaluation.read_bytes()
    result = audit_evaluation(json.loads(data))
    result["evaluation_sha256"] = hashlib.sha256(data).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
