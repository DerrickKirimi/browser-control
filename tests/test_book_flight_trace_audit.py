"""Visible decision checks preserve native outcomes and action-time context."""

from copy import deepcopy

import pytest

from browser_control.book_flight_trace_audit import audit_episode, audit_evaluation

GOAL = "Book the shortest one-way flight from: PIT to: GPT on 10/03/2016."
FIELDS = """
[19] textbox 'From:' value='Pittsburgh, PA (PIT)'
[21] textbox 'To:' value='Biloxi/Gulfport, MS (GPT)'
[25] textbox '' value='10/03/2016'
[27] button 'Search'
"""
FLIGHTS = """
StaticText '8h 3m'
[100] button 'Book flight for $200'
StaticText '1h 2m'
[101] button 'Book flight for $400'
StaticText '1h 2m'
[102] button 'Book flight for $500'
"""


def step(tree, bid, turn=0):
    return {
        "turn": turn,
        "observation": {"accessibility_tree": tree},
        "text": f"<tool_call><function=click><parameter=bid>{bid}</parameter></function></tool_call>",
        "done": False,
        "reward": 0,
    }


def episode(*steps):
    return {
        "goal": GOAL,
        "task_seed": 85,
        "success": False,
        "stop_reason": "horizon",
        "steps": list(steps),
    }


def test_search_checks_action_time_fields_instead_of_prior_correct_milestone():
    typed = FIELDS.replace("Pittsburgh, PA (PIT)", "PIT")
    row = audit_episode(episode(step(FIELDS, "19"), step(typed, "27", 1)))
    assert row["search_with_incorrect_fields"]
    assert row["search_decisions"] == [
        {
            "turn": 1,
            "origin_selected": False,
            "destination_selected": True,
            "date_correct": True,
        }
    ]


@pytest.mark.parametrize("bid,correct", [("100", False), ("101", True), ("102", True)])
def test_duration_criterion_accepts_ties_and_does_not_choose_cheapest(bid, correct):
    row = audit_episode(episode(step(FLIGHTS, bid)))
    assert row["booking_decisions"][0]["matches_visible_criterion"] is correct
    assert row["booking_decisions"][0]["best_visible_bids"] == ["101", "102"]
    assert row["native_success"] is False


def test_cheapest_criterion_and_native_reward_override_no_diagnostic():
    source = episode(step(FLIGHTS, "100"))
    source["goal"] = GOAL.replace("shortest", "cheapest")
    source["steps"][0].update(done=True, reward=1)
    source["success"] = True
    before = deepcopy(source)
    row = audit_episode(source)
    assert row["native_success"]
    assert not row["suboptimal_visible_booking"]
    assert source == before


def test_invalid_output_and_action_errors_are_separate_observations():
    source = episode(step(FIELDS, "27"))
    source["steps"][0].update(text="click('27')", action_error="TimeoutError")
    row = audit_episode(source)
    assert row["invalid_output_turns"] == [0]
    assert row["reported_action_error_turns"] == [0]
    assert not row["search_decisions"]


def test_diagnostics_do_not_replace_native_success_or_ignore_corrupt_reports():
    source = episode(step(FLIGHTS, "100"))
    source["success"] = True
    with pytest.raises(ValueError, match="native terminal reward"):
        audit_episode(source)
    source["success"] = False
    with pytest.raises(ValueError, match="Duplicate"):
        audit_evaluation({"status": "complete", "episodes": [source, source]})
    with pytest.raises(ValueError, match="completed"):
        audit_evaluation({"status": "running", "episodes": [source]})
