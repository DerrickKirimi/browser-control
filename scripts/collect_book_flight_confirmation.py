"""Freeze observed-state probes on a separate development confirmation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing
from pathlib import Path

from browser_control import book_flight_solver
from browser_control.book_flight_diagnostic import CONFIRMATION_SEEDS, build_probes
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing confirmation evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "collecting",
        "cohort": "development-confirmation-50-69",
        "task_seeds": list(CONFIRMATION_SEEDS),
        "solver_sha256": hashlib.sha256(
            Path(book_flight_solver.__file__).read_bytes()
        ).hexdigest(),
        "records": [],
        "probes": [],
    }

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
    try:
        with (
            MiniwobAssetServer(root) as server,
            closing(
                LocalBrowserGymEnvironment.miniwob(
                    "book-flight-nodelay", miniwob_url=server.base_url
                )
            ) as env,
        ):
            for seed in CONFIRMATION_SEEDS:
                record = book_flight_solver.solve(env, seed)
                report["records"].append(record)
                save()
                if not record["success"] or any(
                    step["error"] for step in record["steps"]
                ):
                    raise ValueError(
                        f"Confirmation expert seed {seed} was not clean and successful"
                    )
                report["probes"].extend(build_probes(record))
                save()
                print(seed, "confirmation probes collected", flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:2000])
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
