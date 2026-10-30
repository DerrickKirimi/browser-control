"""Scripted book-flight solver using observed UI controls, never hidden state."""

from __future__ import annotations

import calendar
import re
from dataclasses import asdict

from browser_control.book_flight import (
    BookFlightAction,
    initial_messages,
    observation_text,
)


def parse_goal(goal: str) -> tuple[str, ...]:
    match = re.fullmatch(
        r"Book the (cheapest|shortest) one-way flight from: (.+) to: (.+) on (\d{2})/(\d{2})/(\d{4})\.",
        goal,
    )
    if match is None:
        raise RuntimeError(f"Unexpected goal: {goal}")
    return match.groups()


def best_flight_bids(tree: str, criterion: str) -> list[str]:
    """Bookable flights tied for the requested criterion, read from the tree."""
    candidates = []
    duration = None
    for line in tree.splitlines():
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
        raise RuntimeError(f"No visible flight results: {tree}")
    lowest = min(candidates)[0]
    return [bid for score, bid in sorted(candidates) if score == lowest]


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
        # Case-sensitive like the page's own check: "CHO" must not pick "Anchorage".
        if bid and "StaticText" in line and query in line:
            return bid
    raise RuntimeError(f"No autocomplete item for {query!r}: {tree}")


def solve(env, seed: int) -> dict:
    state = env.reset(seed=seed)
    criterion, origin, destination, month, day, year = parse_goal(
        state.observation.goal
    )
    messages = initial_messages(state.observation)
    record = {"task_seed": seed, "goal": state.observation.goal, "steps": []}

    def act(name: str, *, skill: str, complete=None, **parameters: str) -> None:
        nonlocal state
        action = BookFlightAction(name, parameters)
        for attempt in range(3):
            before = state.observation
            state = env.step(action.render())
            record["steps"].append(
                {
                    "skill": skill,
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

    for label, value, skill in [
        ("From:", origin, "select_origin"),
        ("To:", destination, "select_destination"),
    ]:
        act(
            "fill",
            skill="fill_airport",
            bid=bid_for(state.observation.accessibility_tree, "textbox", label),
            value=value,
        )
        act(
            "click",
            skill=skill,
            bid=autocomplete_bid(state.observation.accessibility_tree, value),
            complete=lambda o: any(
                f"textbox {label!r} value=" in line and ")" in line
                for line in o.accessibility_tree.splitlines()
            ),
        )
    act(
        "click",
        skill="open_calendar",
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
            skill="navigate_calendar",
            bid=bid_for(tree, "link", "Prev" if current > int(month) else "Next"),
            complete=lambda o: f"StaticText '{calendar.month_name[current]}'"
            not in o.accessibility_tree,
        )
    else:
        raise RuntimeError("Calendar navigation exceeded expected bound")
    act(
        "click",
        skill="choose_date",
        bid=bid_for(state.observation.accessibility_tree, "link", str(int(day))),
        complete=lambda o: f"value='{month}/{day}/{year}'" in o.accessibility_tree,
    )
    act(
        "click",
        skill="search",
        bid=bid_for(state.observation.accessibility_tree, "button", "Search"),
        complete=lambda o: "Book flight for $" in o.accessibility_tree,
    )
    best = best_flight_bids(state.observation.accessibility_tree, criterion)
    act("click", skill="book_flight", bid=best[0], complete=lambda o: state.done)
    record.update(
        success=state.done and state.reward > 0, reward=state.reward, messages=messages
    )
    if not record["success"]:
        raise RuntimeError(f"Scripted solver failed: {record}")
    return record
