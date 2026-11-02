"""Prevent duplicate training weights, stale provenance and seed leakage in merges."""

from copy import deepcopy

import pytest

from browser_control.book_flight_corpus import POLICY, merge_expert_corpora
from browser_control.book_flight_sft import expert_examples


def corpus(seed=100):
    record = {
        "task_seed": seed,
        "success": True,
        "reward": 1.0,
        "eligible_for_sft": True,
        "steps": [
            {
                "error": "",
                "observation": {},
                "skill": "choose_date",
                "messages": [{"role": "user", "content": "task"}],
                "assistant": {"role": "assistant", "tool_calls": [{}]},
            }
        ],
    }
    return {
        "status": "complete",
        "task": "book-flight-nodelay",
        "policy": POLICY,
        "solver_sha256": "current",
        "packages": {"browsergym-core": "0.14.3"},
        "seeds": [seed],
        "records": [record],
        "failures": [],
        "examples": expert_examples(record),
    }


def test_merge_preserves_raw_records_and_distinct_example_weights():
    first, second = corpus(), corpus(101)
    expected = deepcopy(first["examples"] + second["examples"])
    merged = merge_expert_corpora([first, second], solver_sha256="current")
    assert merged["examples"] == expected
    assert merged["eligible_episodes"] == 2
    merged["records"][0]["steps"][0]["messages"][0]["content"] = "mutated"
    assert first["records"][0]["steps"][0]["messages"][0]["content"] == "task"


def test_duplicate_seed_rejected_even_if_records_differ():
    first, second = corpus(), corpus()
    second["records"][0]["reward"] = 2.0
    with pytest.raises(ValueError, match="Overlapping"):
        merge_expert_corpora([first, second], solver_sha256="current")


@pytest.mark.parametrize("seed", [30, 10000, 10100, True])
def test_declared_seed_leakage_rejected(seed):
    item = corpus()
    item["seeds"] = [seed]
    with pytest.raises(ValueError, match="training seeds"):
        merge_expert_corpora([item], solver_sha256="current")


@pytest.mark.parametrize(
    "key", ["status", "task", "policy", "solver_sha256", "packages"]
)
def test_mismatched_provenance_rejected(key):
    item = corpus()
    item[key] = "different"
    with pytest.raises(ValueError, match="provenance"):
        merge_expert_corpora([item], solver_sha256="current")


def test_different_browser_versions_cannot_be_combined():
    first, second = corpus(), corpus(101)
    second["packages"]["browsergym-core"] = "another-version"
    with pytest.raises(ValueError, match="provenance differs in packages"):
        merge_expert_corpora([first, second], solver_sha256="current")


def test_deleted_record_rejected_instead_of_silently_losing_coverage():
    item = corpus()
    item["records"] = []
    with pytest.raises(ValueError, match="cover declared"):
        merge_expert_corpora([item], solver_sha256="current")


def test_error_episode_cannot_keep_successful_example():
    item = corpus()
    item["records"][0]["steps"][0]["error"] = "timeout"
    with pytest.raises(ValueError, match="clean successful"):
        merge_expert_corpora([item], solver_sha256="current")
