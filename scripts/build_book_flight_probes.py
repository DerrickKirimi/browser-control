"""Solve development seeds and freeze the teacher-forced diagnostic probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from browser_control.book_flight import BookFlightAction
from browser_control.book_flight_diagnostic import (
    DEVELOPMENT_SEEDS,
    accepted_actions,
    airport_selected,
    build_probes,
)
from browser_control.book_flight_solver import parse_goal, solve
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets

MIN_SOLVED = 19
VALIDATION_SEEDS = (30, 32)
VALIDATED_SKILLS = (
    "select_origin",
    "select_destination",
    "open_calendar",
    "navigate_calendar",
    "choose_date",
    "search",
    "book_flight",
)


def reached(skill: str, goal: str, tree: str, before: str, step) -> bool:
    """Observable outcome each accepted action must produce in the browser."""
    _, origin, destination, month, day, year = parse_goal(goal)
    if skill in ("select_origin", "select_destination"):
        label, place = (
            ("From:", origin) if skill == "select_origin" else ("To:", destination)
        )
        return airport_selected(tree, label, place)
    if skill == "open_calendar":
        return f"StaticText '{year}'" in tree
    if skill == "navigate_calendar":
        months = re.compile(
            r"StaticText '(January|February|March|April|May|June|July|August"
            r"|September|October|November|December)'"
        )
        current = months.search(tree)
        return current is not None and current[1] != months.search(before)[1]
    if skill == "choose_date":
        return f"value='{month}/{day}/{year}'" in tree
    if skill == "search":
        return "Book flight for $" in tree
    return step.done and step.reward > 0


def validate(env, record: dict) -> list[dict]:
    """Replay the solver to each state and execute every accepted action there."""
    rows = []
    seen = set()
    for index, solver_step in enumerate(record["steps"]):
        skill = solver_step["skill"]
        if skill not in VALIDATED_SKILLS or skill in seen:
            continue
        seen.add(skill)
        count = len(
            accepted_actions(
                skill, record["goal"], solver_step["observation"]["accessibility_tree"]
            )
        )
        for position in range(count):
            state = env.reset(seed=record["task_seed"])
            for earlier in record["steps"][:index]:
                state = env.step(earlier["action"])
            before = state.observation.accessibility_tree
            call = accepted_actions(skill, record["goal"], before)[position]
            action = BookFlightAction(call["name"], call["parameters"])
            step = env.step(action.render())
            tree = step.observation.accessibility_tree
            row = {
                "task_seed": record["task_seed"],
                "skill": skill,
                "action": action.render(),
                "reached": reached(skill, record["goal"], tree, before, step),
                "error": (step.observation.last_action_error or "")[:80],
            }
            print(row, flush=True)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lfm3/book-flight/diagnostic/probes.json"),
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve it before building another probe set")
    root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
    records, failures = [], []
    with MiniwobAssetServer(root) as server:
        env = LocalBrowserGymEnvironment.miniwob(
            "book-flight-nodelay", miniwob_url=server.base_url
        )
        try:
            for seed in DEVELOPMENT_SEEDS:
                try:
                    record = solve(env, seed)
                except RuntimeError as error:
                    failures.append({"task_seed": seed, "error": str(error)[:2000]})
                    print(seed, "FAILED", str(error)[:300], flush=True)
                    continue
                records.append(record)
                print(seed, "solved", len(record["steps"]), record["goal"], flush=True)
            validation = []
            if args.validate:
                for record in records:
                    if record["task_seed"] in VALIDATION_SEEDS:
                        validation.extend(validate(env, record))
        finally:
            env.close()
    report = {
        "seeds": list(DEVELOPMENT_SEEDS),
        "solved": len(records),
        "required": MIN_SOLVED,
        "failures": failures,
        "validation": validation,
        "probes": [probe for record in records for probe in build_probes(record)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "solved",
        len(records),
        "probes",
        len(report["probes"]),
        "sha256",
        hashlib.sha256(args.output.read_bytes()).hexdigest(),
    )
    if len(records) < MIN_SOLVED:
        raise SystemExit("solver below the required rate; fix it before any GPU run")


if __name__ == "__main__":
    main()
