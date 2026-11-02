"""Read-only checks of visible search and booking decisions in native traces."""

from __future__ import annotations

import re

from browser_control.actions import InvalidActionError
from browser_control.book_flight import parse_book_flight_action
from browser_control.book_flight_diagnostic import airport_selected
from browser_control.book_flight_solver import best_flight_bids, parse_goal


def audit_episode(episode: dict) -> dict:
    """Describe observed decisions, without assigning rewards or inferring causes.

    Search checks use the state at that action, rather than an earlier milestone.
    A correct visible choice does not prove its click took effect in the browser.
    """
    criterion, origin, destination, month, day, year = parse_goal(episode["goal"])
    if type(episode["success"]) is not bool or not episode["steps"]:
        raise ValueError("Expected native trace with boolean success and steps")
    final = episode["steps"][-1]
    native = bool(final.get("done") and final.get("reward", 0) > 0)
    if native != episode["success"]:
        raise ValueError("Success differs from recorded native terminal reward")
    searches, bookings, invalid, errors = [], [], [], []
    for step in episode["steps"]:
        turn = step["turn"]
        if step.get("action_error"):
            errors.append(turn)
        try:
            action = parse_book_flight_action(step["text"])
        except InvalidActionError:
            invalid.append(turn)
            continue
        if action.name != "click" and not (
            action.name == "press" and action.parameters["key"] == "Enter"
        ):
            continue
        bid = action.parameters["bid"]
        tree = step["observation"]["accessibility_tree"]
        if re.search(rf"\[{re.escape(bid)}\] button 'Search'(?:\s|$)", tree):
            searches.append(
                {
                    "turn": turn,
                    "origin_selected": airport_selected(tree, "From:", origin),
                    "destination_selected": airport_selected(tree, "To:", destination),
                    "date_correct": f"textbox '' value='{month}/{day}/{year}'" in tree,
                }
            )
        visible = re.findall(r"\[([\w-]+)\] button 'Book flight for \$\d+'", tree)
        if bid in visible:
            best = best_flight_bids(tree, criterion)
            bookings.append(
                {
                    "turn": turn,
                    "criterion": criterion,
                    "selected_bid": bid,
                    "best_visible_bids": best,
                    "matches_visible_criterion": bid in best,
                }
            )
    return {
        "task_seed": episode["task_seed"],
        "native_success": native,
        "stop_reason": episode["stop_reason"],
        "search_decisions": searches,
        "booking_decisions": bookings,
        "invalid_output_turns": invalid,
        "reported_action_error_turns": errors,
        "search_with_incorrect_fields": any(
            not all(
                row[key]
                for key in ("origin_selected", "destination_selected", "date_correct")
            )
            for row in searches
        ),
        "suboptimal_visible_booking": any(
            not row["matches_visible_criterion"] for row in bookings
        ),
    }


def audit_evaluation(report: dict) -> dict:
    if report["status"] != "complete":
        raise ValueError("Audit a completed evaluation; preserve partial evidence")
    episodes = [audit_episode(episode) for episode in report["episodes"]]
    seeds = [row["task_seed"] for row in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate episode seeds")
    return {
        "scope": "visible decision audit; flags can overlap and are not causal labels or training rewards",
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "native_successes": sum(row["native_success"] for row in episodes),
            "episodes_with_incorrect_search_fields": sum(
                row["search_with_incorrect_fields"] for row in episodes
            ),
            "episodes_with_suboptimal_visible_booking": sum(
                row["suboptimal_visible_booking"] for row in episodes
            ),
            "episodes_with_reported_action_error": sum(
                bool(row["reported_action_error_turns"]) for row in episodes
            ),
            "episodes_with_invalid_output": sum(
                bool(row["invalid_output_turns"]) for row in episodes
            ),
        },
    }
