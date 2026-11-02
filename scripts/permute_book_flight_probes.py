"""Freeze a consistently renumbered copy of development probes; never overwrite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_control.book_flight_diagnostic import CONFIRMATION_SEEDS, DEVELOPMENT_SEEDS
from browser_control.book_flight_ids import (
    id_permutation,
    relabel_probe,
    structural_ids,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing probe sets; choose a new output path")
    source = json.loads(args.probes.read_text())
    report = {
        "source_sha256": hashlib.sha256(args.probes.read_bytes()).hexdigest(),
        "permutation_seed": args.seed,
        "scheme": "per-probe bijective derangement within character widths; singleton classes fixed",
        "scope": "text-only ID sensitivity test; not executable against original browser IDs",
        "probes": [],
        "mappings": [],
    }
    for index, probe in enumerate(source["probes"]):
        seeds = CONFIRMATION_SEEDS if args.confirmation else DEVELOPMENT_SEEDS
        if probe["task_seed"] not in seeds:
            parser.error("This diagnostic accepts only the fixed development probes")
        mapping = id_permutation(structural_ids(probe), seed=args.seed + index)
        altered = relabel_probe(probe, mapping)
        restored = relabel_probe(altered, {v: k for k, v in mapping.items()})
        if restored != probe:
            raise ValueError(f"ID round trip changed other data in probe {index}")
        target_bids = {
            call["parameters"]["bid"]
            for call in probe["accepted"]
            if "bid" in call["parameters"]
        }
        if any(mapping[bid] == bid for bid in target_bids):
            raise ValueError(f"An accepted target remained unchanged in probe {index}")
        report["probes"].append(altered)
        report["mappings"].append(
            {
                "probe_index": index,
                "mapping": mapping,
                "fixed_ids": [k for k, v in mapping.items() if k == v],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "probes",
        len(report["probes"]),
        "sha256",
        hashlib.sha256(args.output.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()
