"""ID renaming changes structural references, not airport names, dates or prices."""

from copy import deepcopy

import pytest

from browser_control.book_flight_ids import (
    compare_id_scores,
    id_permutation,
    relabel_probe,
    structural_ids,
)


def probe():
    observation = (
        "Goal: Flight 25 from Airport 27 on 10/25/2016, price $34, [bid='25']\n"
        "Last action error: locator('[bid=\"25\"]') timed out\n"
        "Accessibility tree:\n"
        "\t[25] textbox 'Airport 27' value='25'\n"
        "\t[27] button 'Book flight for $34'\n"
        "\t[34] link '25'\n"
        "\t\tStaticText '[25] is literal text, [bid=\"25\"]'\n"
    )
    return {
        "task_seed": 30,
        "skill": "choose_date",
        "goal": "Airport 27 on 10/25/2016",
        "messages": [
            {"role": "user", "content": observation},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "fill",
                            "arguments": {"bid": "25", "value": "Airport 27"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": observation},
        ],
        "accepted": [{"name": "click", "parameters": {"bid": "34"}}],
        "solver_action": "click('34')",
    }


def test_every_structural_reference_changes_consistently():
    original = probe()
    snapshot = deepcopy(original)
    mapping = {"25": "27", "27": "34", "34": "25"}
    result = relabel_probe(original, mapping)
    content = result["messages"][0]["content"]
    assert "[27] textbox 'Airport 27' value='25'" in content
    assert "[34] button 'Book flight for $34'" in content
    assert "[25] link '25'" in content
    assert "locator('[bid=\"27\"]')" in content
    assert result["messages"][1]["tool_calls"][0]["function"]["arguments"] == {
        "bid": "27",
        "value": "Airport 27",
    }
    assert result["accepted"][0]["parameters"]["bid"] == "25"
    assert result["solver_action"] == "click('25')"
    assert original == snapshot


def test_free_text_numbers_and_selector_lookalikes_are_unchanged():
    result = relabel_probe(probe(), {"25": "27", "27": "34", "34": "25"})
    content = result["messages"][0]["content"]
    assert content.splitlines()[0] == probe()["messages"][0]["content"].splitlines()[0]
    assert "StaticText '[25] is literal text, [bid=\"25\"]'" in content
    assert result["goal"] == probe()["goal"]


def test_bijective_mapping_can_be_inverted_without_text_loss():
    mapping = {"25": "27", "27": "34", "34": "25"}
    result = relabel_probe(
        relabel_probe(probe(), mapping), {v: k for k, v in mapping.items()}
    )
    assert result == probe()


def test_permutation_is_reproducible_width_preserving_and_deranged():
    ids = {"1", "2", "25", "27", "34", "123"}
    mapping = id_permutation(ids, seed=23)
    assert mapping == id_permutation(ids, seed=23)
    assert set(mapping.values()) == ids
    assert all(len(key) == len(value) for key, value in mapping.items())
    assert all(key != value for key, value in mapping.items() if key != "123")
    assert mapping["123"] == "123"


def test_extraction_does_not_turn_other_numbers_into_ids():
    assert structural_ids(probe()) == {"25", "27", "34"}


def test_mapping_must_be_complete_and_one_to_one():
    with pytest.raises(ValueError, match="cover"):
        relabel_probe(probe(), {"25": "27"})
    with pytest.raises(ValueError, match="one-to-one"):
        relabel_probe(probe(), {"25": "27", "27": "27", "34": "34"})


@pytest.mark.parametrize("drop,stop", [(3, False), (4, True), (5, True)])
def test_predeclared_four_of_twenty_drop_stops_progression(drop, stop):
    before = {
        "skills": {
            skill: {"count": 20, "correct": 20}
            for skill in (
                "select_origin",
                "select_destination",
                "open_calendar",
                "search",
            )
        }
    }
    after = deepcopy(before)
    after["skills"]["search"]["correct"] -= drop
    result = compare_id_scores(before, after)
    assert result["stop_grpo"] is stop
    assert result["skills"]["search"]["drop"] == drop


def test_incomplete_id_comparison_cannot_pass_gate():
    incomplete = {"skills": {"select_origin": {"count": 19, "correct": 19}}}
    with pytest.raises(ValueError, match="20 matched"):
        compare_id_scores(incomplete, incomplete)
