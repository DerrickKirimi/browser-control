"""Probe scoring, reasoning parsing and the preregistered route decision."""

import pytest

from browser_control.actions import InvalidActionError
from browser_control.book_flight import parse_reasoned_action
from browser_control.book_flight_diagnostic import (
    DECISION_SKILLS,
    THINKING_SYSTEM_PROMPT,
    accepted_actions,
    decide,
    milestones,
    score_probe,
)
from browser_control.book_flight_solver import best_flight_bids

GOAL = "Book the shortest one-way flight from: Thorne Bay, AK to: Palm Springs, CA on 10/03/2016."
CALL = "<tool_call>\n<function=click>\n<parameter=bid>\n34\n</parameter>\n</function>\n</tool_call>"
SUGGESTIONS = """
	[19] textbox 'From:' value='Thorne Bay, AK', focused
		StaticText 'Thorne Bay, AK'
	[21] textbox 'To:'
	[29] list 'Thorne Bay, AK (KTB)'
		[34] listitem ''
			ListMarker ''
				image ''
			StaticText 'Thorne Bay, AK (KTB)'
"""
RESULTS = """
	StaticText '6h 6m'
	[370] button 'Book flight for $449'
	StaticText '0h 2m'
	[392] button 'Book flight for $507'
	StaticText '0h 2m'
	[414] button 'Book flight for $824'
"""


def test_reasoned_output_needs_one_close_and_one_call():
    action = parse_reasoned_action(f"pick the suggestion</think>\n{CALL}")
    assert action.parameters == {"bid": "34"}
    assert action.message()["reasoning_content"] == "pick the suggestion"


@pytest.mark.parametrize(
    "output",
    [
        CALL,
        "still thinking",
        f"a</think>b</think>{CALL}",
        f"reasoning</think>prose {CALL}",
        f"reasoning</think>{CALL}{CALL}",
    ],
)
def test_reasoned_parser_rejects_everything_else(output):
    with pytest.raises(InvalidActionError):
        parse_reasoned_action(output)


def test_strict_parser_still_rejects_reasoning():
    probe = {"accepted": [{"name": "click", "parameters": {"bid": "34"}}]}
    assert score_probe(probe, CALL, "strict")["correct"]
    assert not score_probe(probe, f"thought</think>{CALL}", "strict")["valid"]


def test_thinking_prompt_changes_only_the_no_reasoning_clause():
    assert "no reasoning" not in THINKING_SYSTEM_PROMPT
    assert "exactly one provided tool" in THINKING_SYSTEM_PROMPT


def test_suggestion_probe_accepts_declared_routes_only():
    accepted = accepted_actions("select_origin", GOAL, SUGGESTIONS)
    assert {"name": "click", "parameters": {"bid": "34"}} in accepted
    assert {"name": "click", "parameters": {"bid": "29"}} in accepted
    assert {
        "name": "fill",
        "parameters": {"bid": "19", "value": "Thorne Bay, AK (KTB)"},
    } in accepted
    score = score_probe(
        {"accepted": accepted},
        "<tool_call>\n<function=click>\n<parameter=bid>\n19\n</parameter>\n</function>\n</tool_call>",
        "strict",
    )
    assert score["valid"] and not score["correct"]


def test_airport_codes_match_case_sensitively_like_the_page():
    tree = """
	[34] listitem ''
		StaticText 'Anchorage, AK (ANC)'
	[35] listitem ''
		StaticText 'Charlottesville, VA (CHO)'
"""
    goal = "Book the cheapest one-way flight from: CHO to: IAH on 11/07/2016."
    accepted = accepted_actions("select_origin", goal, f"[19] textbox 'From:'\n{tree}")
    assert {"name": "click", "parameters": {"bid": "35"}} in accepted
    assert all(call["parameters"].get("key") != "ArrowDown" for call in accepted)


def test_book_probe_accepts_every_tied_best_flight():
    assert best_flight_bids(RESULTS, "shortest") == ["392", "414"]
    assert best_flight_bids(RESULTS, "cheapest") == ["370"]


def _episode(*trees, success=False):
    return {
        "goal": GOAL,
        "success": success,
        "steps": [{"observation": {"accessibility_tree": tree}} for tree in trees[:-1]],
        "final_observation": {"accessibility_tree": trees[-1]},
    }


def test_typed_airport_is_not_a_selected_milestone():
    typed = milestones(_episode(SUGGESTIONS))
    assert not typed["origin_selected"]
    chosen = milestones(
        _episode(
            "[19] textbox 'From:' value='Thorne Bay, AK (KTB)'\n"
            "[21] textbox 'To:' value=\"Palm Springs, CA (PSP)\"\n"
            "[25] textbox '' value='10/03/2016'\n",
            "[370] button 'Book flight for $449'",
        )
    )
    assert chosen == {
        "origin_selected": True,
        "destination_selected": True,
        "date_correct": True,
        "results_shown": True,
        "success": False,
    }


def _cell(valid=0.9, skill=0.9, successes=0, first_turn=7):
    return {
        "probes": {
            "decision_skills": {"valid_rate": valid},
            "skills": {name: {"correct_rate": skill} for name in DECISION_SKILLS},
        },
        "rollouts": {"successes": successes, "first_turn_valid": first_turn},
    }


@pytest.mark.parametrize(
    ("c0", "c2", "route"),
    [
        (_cell(first_turn=1), _cell(), "stop-debug-harness"),
        (_cell(first_turn=13), _cell(), "stop-debug-harness"),
        (_cell(valid=0.5), _cell(valid=0.79), "format-sft-then-repeat"),
        (_cell(skill=0.5), _cell(valid=0.5), "expert-sft-warm-start"),
        (_cell(), _cell(valid=0.5), "grpo-smoke-milestone-reward"),
        (_cell(skill=0.5), _cell(), "grpo-smoke-milestone-reward"),
        (_cell(), _cell(successes=2), "new-frozen-gate-10100-10199"),
    ],
)
def test_decision_rule_order(c0, c2, route):
    assert decide({"C0": c0, "C2": c2})["route"] == route


def test_any_decision_skill_below_threshold_blocks_rollout_routes():
    cell = _cell(successes=5)
    cell["probes"]["skills"]["book_flight"]["correct_rate"] = 0.65
    assert decide({"C0": cell, "C2": _cell(valid=0.1)})["route"] == (
        "expert-sft-warm-start"
    )


def test_decision_waits_for_both_sampled_cells():
    assert decide({"C0": _cell()})["route"] == "incomplete"


def test_ties_prefer_the_cheaper_mode():
    assert decide({"C0": _cell(), "C2": _cell()})["cell"] == "C0"
