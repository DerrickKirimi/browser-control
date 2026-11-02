"""Solve book-flight through observed UI controls, without inspecting hidden state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from browser_control.book_flight_solver import solve
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/lfm3/book-flight/scripted.json")
    )
    args = parser.parse_args()
    assert 1 <= args.episodes <= 10, "Only development seeds 0..9 are allowed here"
    root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with MiniwobAssetServer(root) as server:
        for task in ("book-flight-nodelay", "book-flight"):
            env = LocalBrowserGymEnvironment.miniwob(task, miniwob_url=server.base_url)
            try:
                for seed in range(args.episodes):
                    record = solve(env, seed)
                    record["task"] = task
                    records.append(record)
                    args.output.write_text(json.dumps(records, indent=2) + "\n")
                    print(
                        task,
                        seed,
                        record["success"],
                        len(record["steps"]),
                        record["goal"],
                        flush=True,
                    )
            finally:
                env.close()


if __name__ == "__main__":
    main()
