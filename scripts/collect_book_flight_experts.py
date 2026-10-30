"""Collect successful, error-free solver conversations on training seeds only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from browser_control import book_flight_solver
from browser_control.book_flight_sft import TRAINING_SEEDS, expert_examples
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = range(args.start_seed, args.start_seed + args.episodes)
    if not seeds or any(seed not in TRAINING_SEEDS for seed in seeds):
        parser.error("Only training seeds 100..9999 are permitted")
    if args.output.exists():
        parser.error("Preserve the existing corpus; choose another path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
    report = {
        "status": "collecting",
        "task": "book-flight-nodelay",
        "seeds": list(seeds),
        "records": [],
        "failures": [],
        "examples": [],
        "policy": "exclude-entire-episode-on-any-action-error",
        "solver_sha256": hashlib.sha256(
            Path(book_flight_solver.__file__).read_bytes()
        ).hexdigest(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("browsergym-core", "browsergym-miniwob", "playwright")
        },
    }
    with MiniwobAssetServer(root) as server:
        env = LocalBrowserGymEnvironment.miniwob(
            "book-flight-nodelay", miniwob_url=server.base_url
        )
        try:
            for seed in seeds:
                try:
                    record = book_flight_solver.solve(env, seed)
                    examples = expert_examples(record)
                    record["eligible_for_sft"] = bool(examples)
                    report["records"].append(record)
                    report["examples"].extend(examples)
                    print(seed, "solved", "clean_examples", len(examples), flush=True)
                except RuntimeError as error:
                    report["failures"].append(
                        {"task_seed": seed, "error": str(error)[:2000]}
                    )
                    print(seed, "FAILED", str(error)[:200], flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            env.close()
    report["status"] = "complete"
    report["eligible_episodes"] = sum(r["eligible_for_sft"] for r in report["records"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["examples"]:
        raise SystemExit("No error-free expert examples collected")


if __name__ == "__main__":
    main()
