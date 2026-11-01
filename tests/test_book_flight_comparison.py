"""Comparison readings require matching tasks, settings and native outcomes."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from browser_control.book_flight_comparison import compare_adapter_evaluations
from browser_control.book_flight_diagnostic import COMPARISON_SEEDS, SAMPLING_128


def evaluation(successes=range(70, 90)):
    return {
        "status": "complete",
        "manifest": {
            "config": {
                "model_revision": "pinned",
                "decoder": asdict(SAMPLING_128),
                "generation_seed": "task_seed * 16 + turn",
                "task_seeds": list(COMPARISON_SEEDS),
                "attention": "sdpa",
                "max_prompt_tokens": 16384,
                "prefill_chunk_size": 512,
                "max_steps": 15,
                "enable_thinking": False,
            }
        },
        "episodes": [
            {
                "task_seed": seed,
                "goal": f"goal{seed}",
                "success": seed in successes,
                "steps": [
                    {
                        "turn": 0,
                        "generation_seed": seed * 16,
                        "observation": {"tree": f"tree{seed}"},
                        "action": "click('1')",
                        "done": True,
                        "reward": float(seed in successes),
                    }
                ],
            }
            for seed in COMPARISON_SEEDS
        ],
    }


def test_recomputes_pairs_instead_of_trusting_supplied_summary():
    before = evaluation()
    after = evaluation((set(range(70, 90)) - {70}) | {91, 92})
    after["summary"] = {"successes": 999}
    result = compare_adapter_evaluations(before, after)
    assert (result["baseline_successes"], result["candidate_successes"]) == (20, 21)
    assert result["wins"] == [91, 92] and result["losses"] == [70]
    assert result["net_wins"] == 1
    assert result["reading"] == "no-detectable-effect-at-this-budget"


@pytest.mark.parametrize(
    "net,reading",
    [
        (4, "candidate-for-replication"),
        (-4, "harmful-at-these-settings"),
        (3, "no-detectable-effect-at-this-budget"),
    ],
)
def test_declared_reading_boundaries(net, reading):
    start = set(range(70, 80))
    end = (
        start | set(range(80, 80 + net))
        if net > 0
        else start - set(range(70, 70 - net))
    )
    assert (
        compare_adapter_evaluations(evaluation(start), evaluation(end))["reading"]
        == reading
    )


def test_mismatched_decoder_rejected():
    before, after = evaluation(), evaluation()
    after["manifest"]["config"]["decoder"]["temperature"] = 1.25
    with pytest.raises(ValueError, match="settings differ: decoder"):
        compare_adapter_evaluations(before, after)


def test_missing_seed_rejected():
    after = evaluation()
    after["episodes"].pop()
    with pytest.raises(ValueError, match="complete ordered"):
        compare_adapter_evaluations(evaluation(), after)


def test_changed_initial_state_rejected():
    after = evaluation()
    after["episodes"][0]["steps"][0]["observation"] = {"tree": "different"}
    with pytest.raises(ValueError, match="Initial task state differs"):
        compare_adapter_evaluations(evaluation(), after)


def test_changed_generation_seed_rejected():
    after = evaluation()
    after["episodes"][0]["steps"][0]["generation_seed"] += 1
    with pytest.raises(ValueError, match="Generation seeds differ"):
        compare_adapter_evaluations(evaluation(), after)


def test_success_must_match_native_terminal_reward():
    after = evaluation()
    after["episodes"][0]["steps"][0]["reward"] = 0.0
    with pytest.raises(ValueError, match="native terminal reward"):
        compare_adapter_evaluations(evaluation(), after)


def test_legacy_label_needs_explicit_settings_and_does_not_override_new_values():
    before, after = evaluation(), evaluation()
    settings = deepcopy(before["manifest"]["config"])
    settings["decoder_name"] = "SAMPLING_128"
    before["manifest"]["config"] = {
        "model_revision": "pinned",
        "decoder": "SAMPLING_128",
        "task_seeds": list(COMPARISON_SEEDS),
        "generation_seed": "task_seed * 16 + turn",
    }
    with pytest.raises(ValueError, match="explicitly resolved"):
        compare_adapter_evaluations(before, after)
    assert (
        compare_adapter_evaluations(before, after, legacy_settings=settings)["net_wins"]
        == 0
    )
    after["manifest"]["config"]["decoder"]["temperature"] = 1.25
    with pytest.raises(ValueError, match="settings differ"):
        compare_adapter_evaluations(before, after, legacy_settings=settings)
