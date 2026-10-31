"""Focused passes are explicit and cannot silently mix evaluation data or adapters."""

from copy import deepcopy

import pytest

from browser_control.book_flight_curriculum import (
    plan_examples,
    validate_initial_adapter,
)


def examples():
    return [
        {"task_seed": 100, "turn": n, "skill": skill, "messages": []}
        for n, skill in enumerate(["choose_date", "book_flight", "search"])
    ]


def test_selected_skills_repeat_exactly_the_declared_number_of_passes():
    source = examples()
    before = deepcopy(source)
    planned = plan_examples(source, skills=["choose_date", "book_flight"], passes=4)
    assert len(planned) == 8
    for index in range(4):
        assert {row["turn"] for row in planned if row["pass_index"] == index} == {0, 1}
    assert source == before
    assert planned == plan_examples(
        source, skills=["choose_date", "book_flight"], passes=4
    )


def test_default_pass_does_not_repeat_examples():
    planned = plan_examples(examples())
    assert len(planned) == len({row["turn"] for row in planned}) == 3


@pytest.mark.parametrize(
    "kwargs", [{"passes": 0}, {"skills": []}, {"skills": ["unknown"]}]
)
def test_invalid_curriculum_is_rejected(kwargs):
    with pytest.raises(ValueError):
        plan_examples(examples(), **kwargs)


def test_reserved_data_is_rejected():
    with pytest.raises(ValueError, match="training seeds"):
        plan_examples([{**examples()[0], "task_seed": 10100}])


def test_focus_replay_preserves_declared_anchor_budget():
    planned = plan_examples(
        examples(), skills=["choose_date", "book_flight"], passes=4, replay_per_skill=1
    )
    assert len(planned) == 12
    assert sum(row["curriculum_role"] == "replay" for row in planned) == 4
    assert {row["skill"] for row in planned if row["curriculum_role"] == "replay"} == {
        "search"
    }


def test_replay_cannot_exceed_available_pool():
    with pytest.raises(ValueError, match="Replay budget"):
        plan_examples(
            examples(), skills=["choose_date", "book_flight"], replay_per_skill=2
        )


def adapter():
    expected = {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj"],
    }
    return expected, {
        **deepcopy(expected),
        "base_model_name_or_path": "fixture",
        "revision": "pin",
    }


def test_initial_adapter_contract_accepts_matching_pinned_parameters():
    expected, config = adapter()
    validate_initial_adapter(config, expected, revision="pin", model_id="fixture")


@pytest.mark.parametrize(
    "key,value",
    [
        ("r", 8),
        ("revision", "other"),
        ("use_dora", True),
        ("rank_pattern", {"q_proj": 2}),
        ("base_model_name_or_path", "other"),
    ],
)
def test_initial_adapter_contract_rejects_config_drift(key, value):
    expected, config = adapter()
    config[key] = value
    with pytest.raises(ValueError):
        validate_initial_adapter(config, expected, revision="pin", model_id="fixture")
