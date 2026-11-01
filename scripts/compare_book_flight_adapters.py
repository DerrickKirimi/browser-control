"""Recompute a matched native development comparison from two saved evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_control.book_flight_comparison import compare_adapter_evaluations
from browser_control.book_flight_frozen import compare_frozen_evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--legacy-settings", type=Path)
    parser.add_argument("--frozen-protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous comparison evidence")
    if args.frozen_protocol and args.legacy_settings:
        parser.error("Frozen comparisons require full recorded settings")
    inputs = {"baseline": args.baseline, "candidate": args.candidate}
    if args.legacy_settings:
        inputs["legacy_settings"] = args.legacy_settings
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    if args.frozen_protocol:
        inputs["frozen_protocol"] = args.frozen_protocol
        protocol_bytes = args.frozen_protocol.read_bytes()
        payload = compare_frozen_evaluations(
            baseline,
            candidate,
            json.loads(protocol_bytes),
            hashlib.sha256(protocol_bytes).hexdigest(),
        )
    else:
        payload = compare_adapter_evaluations(
            baseline,
            candidate,
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
            )
        }
    )


if __name__ == "__main__":
    main()
