"""Development-only probes, milestones and route decision for book-flight modes."""

from __future__ import annotations

import re
from typing import Any

from browser_control.actions import InvalidActionError
from browser_control.book_flight import (
    SYSTEM_PROMPT,
    parse_book_flight_action,
    parse_reasoned_action,
)
from browser_control.book_flight_solver import (
    autocomplete_bid,
    best_flight_bids,
    bid_for,
    parse_goal,
)
from browser_control.transformers_policy import DecoderConfig

DEVELOPMENT_SEEDS = range(30, 50)
CONFIRMATION_SEEDS = range(50, 70)
COMPARISON_SEEDS = range(70, 100)
DECISION_SKILLS = (
    "select_origin",
    "select_destination",
    "open_calendar",
    "choose_date",
    "search",
    "book_flight",
)
# Reported only: the decision rule was fixed on the six skills above.
REPORT_ONLY_SKILLS = ("navigate_calendar",)
MIN_VALID_RATE = 0.8
MIN_SKILL_RATE = 0.7
MIN_ROLLOUT_SUCCESSES = 2
FIRST_TURN_VALID_RANGE = (2, 12)

THINKING_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Return only the tool call, with no reasoning or prose.",
    "After thinking, return only the tool call, with no prose.",
)
assert THINKING_SYSTEM_PROMPT != SYSTEM_PROMPT

SAMPLING_128 = DecoderConfig("sampling", True, 128, temperature=1.0, top_p=0.95)
CELLS: dict[str, dict[str, Any]] = {
    # C0 repeats the frozen gate's mode; C1 is reported only as the frozen baseline.
    "C0": {
        "decoder": SAMPLING_128,
        "enable_thinking": False,
        "parser": "strict",
        "system_prompt": SYSTEM_PROMPT,
    },
    "C2": {
        "decoder": DecoderConfig("sampling", True, 512, temperature=1.0, top_p=0.95),
        "enable_thinking": True,
        "parser": "one-call-after-think",
        "system_prompt": THINKING_SYSTEM_PROMPT,
    },
    "C1": {
        "decoder": DecoderConfig("greedy", False, 128),
        "enable_thinking": False,
        "parser": "strict",
        "system_prompt": SYSTEM_PROMPT,
    },
}
PARSERS = {
    "strict": parse_book_flight_action,
    "one-call-after-think": parse_reasoned_action,
}


def _call(name: str, **parameters: str) -> dict[str, Any]:
    return {"name": name, "parameters": parameters}


def _activate(bid: str) -> list[dict[str, Any]]:
    return [_call("click", bid=bid), _call("press", bid=bid, key="Enter")]


def _suggestion_text(tree: str, item_bid: str) -> str:
    lines = tree.splitlines()
    start = next(
        index for index, line in enumerate(lines) if f"[{item_bid}] listitem" in line
    )
    for line in lines[start:]:
        found = re.search(r"StaticText (['\"])(.+)\1", line)
        if found:
            return found[2]
    raise RuntimeError(f"No suggestion text for {item_bid}: {tree}")


def accepted_actions(skill: str, goal: str, tree: str) -> list[dict[str, Any]]:
    """Actions declared correct at a solver state, before any model output exists.

    `scripts/build_book_flight_probes.py --validate` executes each one in the
    browser and checks that it reaches the skill's observable outcome.
    """
    criterion, origin, destination, month, day, _ = parse_goal(goal)
    if skill in ("select_origin", "select_destination"):
        label, value = (
            ("From:", origin) if skill == "select_origin" else ("To:", destination)
        )
        box = bid_for(tree, "textbox", label)
        item = autocomplete_bid(tree, value)
        accepted = [
            _call("click", bid=item),
            _call("fill", bid=box, value=_suggestion_text(tree, item)),
        ]
        # Enter on a suggestion does not select it; ArrowDown takes the first one.
        first = re.search(r"\[([\w-]+)\] listitem", tree)
        if first and first[1] == item:
            accepted.append(_call("press", bid=box, key="ArrowDown"))
        # Clicking a one-item list lands on its only suggestion.
        parent = re.search(r"\[([\w-]+)\] list '", tree)
        if parent and len(re.findall(r"\] listitem", tree)) == 1:
            accepted.append(_call("click", bid=parent[1]))
        return accepted
    if skill == "open_calendar":
        # Enter on the read-only field does not open the calendar.
        return [_call("click", bid=bid_for(tree, "textbox", ""))]
    if skill == "navigate_calendar":
        months = (
            "January February March April May June July August September "
            "October November December"
        ).split()
        current = next(
            number
            for number, name in enumerate(months, 1)
            if f"StaticText '{name}'" in tree
        )
        # Enter on Prev/Next does not change the month.
        link = "Prev" if current > int(month) else "Next"
        return [_call("click", bid=bid_for(tree, "link", link))]
    if skill == "choose_date":
        return _activate(bid_for(tree, "link", str(int(day))))
    if skill == "search":
        return _activate(bid_for(tree, "button", "Search"))
    if skill == "book_flight":
        return [
            call for bid in best_flight_bids(tree, criterion) for call in _activate(bid)
        ]
    raise ValueError(f"unknown skill: {skill}")


def build_probes(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The first attempt at each skill in a solved episode is that skill's probe."""
    probes: dict[str, dict[str, Any]] = {}
    for step in record["steps"]:
        skill = step["skill"]
        if skill in (*DECISION_SKILLS, *REPORT_ONLY_SKILLS) and skill not in probes:
            tree = step["observation"]["accessibility_tree"]
            probes[skill] = {
                "task_seed": record["task_seed"],
                "skill": skill,
                "goal": record["goal"],
                "messages": step["messages"],
                "solver_action": step["action"],
                "prior_error": bool(step["observation"]["last_action_error"]),
                "accepted": accepted_actions(skill, record["goal"], tree),
            }
    missing = set(DECISION_SKILLS) - set(probes)
    if missing:
        raise RuntimeError(f"Seed {record['task_seed']} lacks probes: {missing}")
    return list(probes.values())


def score_probe(probe: dict[str, Any], output: str, parser: str) -> dict[str, Any]:
    try:
        action = PARSERS[parser](output)
    except InvalidActionError as error:
        return {"valid": False, "correct": False, "error": str(error)}
    call = {"name": action.name, "parameters": action.parameters}
    return {"valid": True, "correct": call in probe["accepted"], "action": call}


def airport_selected(tree: str, label: str, place: str) -> bool:
    """A catalogue value holds the goal's text, the page's own correctness check."""
    return any(
        place in found[2] and re.search(r"\([A-Z0-9]{3}\)$", found[2])
        for found in re.finditer(
            rf"textbox {re.escape(repr(label))} value=(['\"])(.*?)\1(?:,|\s|$)", tree
        )
    )


def milestones(episode: dict[str, Any]) -> dict[str, bool]:
    """Progress visible in observations; native success stays the reported metric."""
    _, origin, destination, month, day, year = parse_goal(episode["goal"])
    trees = [
        *(step["observation"]["accessibility_tree"] for step in episode["steps"]),
        episode["final_observation"]["accessibility_tree"],
    ]

    return {
        "origin_selected": any(
            airport_selected(tree, "From:", origin) for tree in trees
        ),
        "destination_selected": any(
            airport_selected(tree, "To:", destination) for tree in trees
        ),
        "date_correct": any(
            f"textbox '' value='{month}/{day}/{year}'" in tree for tree in trees
        ),
        "results_shown": any("Book flight for $" in tree for tree in trees),
        "success": bool(episode["success"]),
    }


def summarize_probes(results: list[dict[str, Any]]) -> dict[str, Any]:
    def rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        return {
            "count": count,
            "valid": sum(row["valid"] for row in rows),
            "correct": sum(row["correct"] for row in rows),
            "valid_rate": sum(row["valid"] for row in rows) / count if count else None,
            "correct_rate": sum(row["correct"] for row in rows) / count
            if count
            else None,
        }

    decision = [row for row in results if row["skill"] in DECISION_SKILLS]
    return {
        "decision_skills": rates(decision),
        "after_prior_error": rates([row for row in decision if row["prior_error"]]),
        "without_prior_error": rates(
            [row for row in decision if not row["prior_error"]]
        ),
        "skills": {
            skill: rates([row for row in results if row["skill"] == skill])
            for skill in (*DECISION_SKILLS, *REPORT_ONLY_SKILLS)
        },
    }


def summarize_rollouts(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [step for episode in episodes for step in episode["steps"]]
    first = [episode["steps"][0] for episode in episodes if episode["steps"]]
    reached = [milestones(episode) for episode in episodes]
    reasons: dict[str, int] = {}
    for episode in episodes:
        reasons[episode["stop_reason"]] = reasons.get(episode["stop_reason"], 0) + 1
    return {
        "episodes": len(episodes),
        "successes": sum(episode["success"] for episode in episodes),
        "first_turn_valid": sum(bool(step.get("valid_action")) for step in first),
        "turns": len(turns),
        "valid_turns": sum(bool(step.get("valid_action")) for step in turns),
        "action_errors": sum(episode["action_errors"] for episode in episodes),
        "stop_reasons": reasons,
        "milestones": {
            name: sum(row[name] for row in reached)
            for name in (reached[0] if reached else ())
        },
        "max_prompt_tokens": max(
            (step.get("prompt_tokens", 0) for step in turns), default=0
        ),
    }


def _cell_route(cell: dict[str, Any]) -> str:
    skills = cell["probes"]["skills"]
    if any(skills[skill]["correct_rate"] < MIN_SKILL_RATE for skill in DECISION_SKILLS):
        return "expert-sft-warm-start"
    if cell["rollouts"]["successes"] >= MIN_ROLLOUT_SUCCESSES:
        return "new-frozen-gate-10100-10199"
    return "grpo-smoke-milestone-reward"


ROUTE_RANK = {
    "new-frozen-gate-10100-10199": 2,
    "grpo-smoke-milestone-reward": 1,
    "expert-sft-warm-start": 0,
}


def decide(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the decision rule in order, on the sampled cells C0 and C2.

    The rule names autocomplete and calendar-opening for its SFT branch but
    leaves any other skill below threshold undefined; any decision skill below it
    takes that branch here. Between qualifying cells, the route furthest from SFT
    wins, then rollout successes, then the weakest skill, then C0 as the cheaper mode.
    """
    if not {"C0", "C2"} <= set(summaries):
        return {"route": "incomplete", "reason": "C0 and C2 are both required"}
    low, high = FIRST_TURN_VALID_RANGE
    first_turn = summaries["C0"]["rollouts"]["first_turn_valid"]
    if not low <= first_turn <= high:
        return {
            "route": "stop-debug-harness",
            "reason": f"C0 first-turn validity {first_turn}/20 outside {low}-{high}",
        }
    routes = {}
    for name in ("C0", "C2"):
        cell = summaries[name]
        if cell["probes"]["decision_skills"]["valid_rate"] >= MIN_VALID_RATE:
            skills = cell["probes"]["skills"]
            routes[name] = {
                "route": _cell_route(cell),
                "successes": cell["rollouts"]["successes"],
                "weakest_skill": min(
                    skills[skill]["correct_rate"] for skill in DECISION_SKILLS
                ),
            }
    if not routes:
        return {
            "route": "format-sft-then-repeat",
            "reason": f"probe validity below {MIN_VALID_RATE:.0%} in C0 and C2",
        }
    chosen = max(
        routes,
        key=lambda name: (
            ROUTE_RANK[routes[name]["route"]],
            routes[name]["successes"],
            routes[name]["weakest_skill"],
            name == "C0",
        ),
    )
    return {"route": routes[chosen]["route"], "cell": chosen, "qualifying": routes}
