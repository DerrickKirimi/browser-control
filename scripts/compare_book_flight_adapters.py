"""Recompute a matched native development comparison from two saved evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_control.book_flight_comparison import compare_adapter_evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--legacy-settings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous comparison evidence")
    inputs = {"baseline": args.baseline, "candidate": args.candidate}
    if args.legacy_settings:
        inputs["legacy_settings"] = args.legacy_settings
    payload = compare_adapter_evaluations(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        legacy_settings=json.loads(args.legacy_settings.read_text())
        if args.legacy_settings
        else None,
    )
    payload["inputs_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in inputs.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        {
            key: payload[key]
            for key in (
                "baseline_successes",
                "candidate_successes",
                "wins",
                "losses",
                "reading",
            )
        }
    )


if __name__ == "__main__":
    main()
