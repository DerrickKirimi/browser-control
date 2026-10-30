"""Solve book-flight through observed UI controls, without inspecting hidden state."""

from __future__ import annotations

import argparse
import calendar
import json
import re
from dataclasses import asdict
from pathlib import Path

from browser_control.book_flight import (
    BookFlightAction,
    initial_messages,
    observation_text,
)
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets


def bid_for(tree: str, role: str, label: str) -> str:
    match = re.search(rf"\[([\w-]+)\] {role} {re.escape(repr(label))}(?:\s|$)", tree)
    if match is None:
        raise RuntimeError(f"No {role} {label!r} in observation: {tree}")
    return match[1]


def autocomplete_bid(tree: str, query: str) -> str:
    bid = None
    for line in tree.splitlines():
        item = re.search(r"\[([\w-]+)\] listitem", line)
        if item:
            bid = item[1]
        if bid and "StaticText" in line and query.casefold() in line.casefold():
            return bid
    raise RuntimeError(f"No autocomplete item for {query!r}: {tree}")


def solve(env, seed: int) -> dict:
    state = env.reset(seed=seed)
    match = re.fullmatch(
        r"Book the (cheapest|shortest) one-way flight from: (.+) to: (.+) on (\d{2})/(\d{2})/(\d{4})\.",
        state.observation.goal,
    )
    if match is None:
        raise RuntimeError(f"Unexpected goal: {state.observation.goal}")
    criterion, origin, destination, month, day, year = match.groups()
    messages = initial_messages(state.observation)
    record = {"task_seed": seed, "goal": state.observation.goal, "steps": []}

    def act(name: str, *, complete=None, **parameters: str) -> None:
        nonlocal state
        action = BookFlightAction(name, parameters)
        for attempt in range(3):
            before = state.observation
            state = env.step(action.render())
            record["steps"].append(
                {
                    "action": action.render(),
                    "observation": asdict(before),
                    "messages": list(messages),
                    "reward": state.reward,
                    "done": state.done,
                    "error": state.observation.last_action_error,
                }
            )
            messages.append(action.message())
            if not state.done:
                messages.append(
                    {"role": "tool", "content": observation_text(state.observation)}
                )
            if state.done or not state.observation.last_action_error:
                return
            if complete is not None and complete(state.observation):
                return  # The click took effect despite a timeout; do not repeat it.
        raise RuntimeError(f"{action.render()}: {state.observation.last_action_error}")

    for label, value in [("From:", origin), ("To:", destination)]:
        act(
            "fill",
            bid=bid_for(state.observation.accessibility_tree, "textbox", label),
            value=value,
        )
        act(
            "click",
            bid=autocomplete_bid(state.observation.accessibility_tree, value),
            complete=lambda o: any(
                f"textbox {label!r} value=" in line and ")" in line
                for line in o.accessibility_tree.splitlines()
            ),
        )
    act(
        "click",
        bid=bid_for(state.observation.accessibility_tree, "textbox", ""),
        complete=lambda o: f"StaticText '{year}'" in o.accessibility_tree,
    )
    for _ in range(4):
        tree = state.observation.accessibility_tree
        current = next(
            (
                m
                for m in range(1, 13)
                if f"StaticText '{calendar.month_name[m]}'" in tree
            ),
            None,
        )
        if current is None or f"StaticText '{year}'" not in tree:
            raise RuntimeError(f"Unknown calendar state: {tree}")
        if current == int(month):
            break
        act(
            "click",
            bid=bid_for(tree, "link", "Prev" if current > int(month) else "Next"),
            complete=lambda o: f"StaticText '{calendar.month_name[current]}'"
            not in o.accessibility_tree,
        )
    else:
        raise RuntimeError("Calendar navigation exceeded expected bound")
    act(
        "click",
        bid=bid_for(state.observation.accessibility_tree, "link", str(int(day))),
        complete=lambda o: f"value='{month}/{day}/{year}'" in o.accessibility_tree,
    )
    act(
        "click",
        bid=bid_for(state.observation.accessibility_tree, "button", "Search"),
        complete=lambda o: "Book flight for $" in o.accessibility_tree,
    )
    candidates = []
    duration = None
    for line in state.observation.accessibility_tree.splitlines():
        found = re.search(r"StaticText '(\d+)h (\d+)m'", line)
        if found:
            duration = int(found[1]) * 60 + int(found[2])
        button = re.search(r"\[([\w-]+)\] button 'Book flight for \$(\d+)'", line)
        if button:
            assert duration is not None
            candidates.append(
                (int(button[2]) if criterion == "cheapest" else duration, button[1])
            )
    if not candidates:
        raise RuntimeError(
            f"No visible flight results: {state.observation.accessibility_tree}"
        )
    act("click", bid=min(candidates)[1], complete=lambda o: state.done)
    record.update(
        success=state.done and state.reward > 0, reward=state.reward, messages=messages
    )
    if not record["success"]:
        raise RuntimeError(f"Scripted solver failed: {record}")
    return record


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
