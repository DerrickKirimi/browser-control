"""Reserved evaluations require locked policies, settings and complete native pairs."""

from copy import deepcopy

import pytest

from browser_control.book_flight_frozen import (
    FROZEN_SETTINGS,
    RESERVED_SEEDS,
    compare_frozen_evaluations,
    validate_frozen_arm,
    validate_frozen_protocol,
)


def protocol():
    return {
        "schema": "book-flight-frozen-v1",
        "task_seeds": list(RESERVED_SEEDS),
        "settings": deepcopy(FROZEN_SETTINGS),
        "arms": {"sft": "a" * 64, "sft_grpo": "b" * 64},
    }


def evaluation(arm, successes):
    locked = protocol()
    return {
        "status": "complete",
        "manifest": {
            "config": {
                **deepcopy(FROZEN_SETTINGS),
                "task_seeds": list(RESERVED_SEEDS),
                "adapter_sha256": locked["arms"][arm],
                "frozen_arm": arm,
                "frozen_protocol_sha256": "c" * 64,
            }
        },
        "episodes": [
            {
                "task_seed": seed,
                "goal": str(seed),
                "success": seed in successes,
                "steps": [
                    {
                        "turn": 0,
                        "generation_seed": seed * 16,
                        "observation": {"tree": str(seed)},
                        "action": "click('1')",
                        "done": True,
                        "reward": float(seed in successes),
                    }
                ],
            }
            for seed in RESERVED_SEEDS
        ],
    }


def test_reserved_pairs_are_recomputed_without_development_reading_rule():
    before = evaluation("sft", range(10100, 10200))
    after = evaluation("sft_grpo", range(10101, 10205))
    after["summary"] = {"successes": 999}
    result = compare_frozen_evaluations(before, after, protocol(), "c" * 64)
    assert (result["baseline_successes"], result["candidate_successes"]) == (100, 104)
    assert result["wins"] == list(range(10200, 10205))
    assert result["losses"] == [10100]
    assert result["success_rate_difference"] == 0.02
    assert result["baseline_wilson_95"] == pytest.approx((0.43136, 0.56864), abs=1e-5)
    assert "reading" not in result


@pytest.mark.parametrize("change", ["seed", "setting", "hash", "missing_arm"])
def test_protocol_rejects_unlocked_inputs(change):
    locked = protocol()
    if change == "seed":
        locked["task_seeds"][0] = 70
    elif change == "setting":
        locked["settings"]["decoder"]["temperature"] = 1.25
    elif change == "hash":
        locked["arms"]["sft"] = "unselected"
    else:
        del locked["arms"]["sft_grpo"]
    with pytest.raises(ValueError):
        validate_frozen_protocol(locked)


def test_arm_validation_refuses_substituted_weights():
    validate_frozen_arm(protocol(), "sft", "a" * 64)
    with pytest.raises(ValueError, match="locked policy"):
        validate_frozen_arm(protocol(), "sft", "b" * 64)


@pytest.mark.parametrize("change", ["protocol", "arm", "weights", "cohort", "native"])
def test_reports_cannot_be_relabelled_to_enter_reserved_comparison(change):
    before = evaluation("sft", range(10100, 10200))
    after = evaluation("sft_grpo", range(10100, 10200))
    config = after["manifest"]["config"]
    if change == "protocol":
        config["frozen_protocol_sha256"] = "d" * 64
    elif change == "arm":
        config["frozen_arm"] = "sft"
    elif change == "weights":
        config["adapter_sha256"] = "a" * 64
    elif change == "cohort":
        after["episodes"].pop()
    else:
        after["episodes"][0]["steps"][0]["reward"] = 0.0
    with pytest.raises(ValueError):
        compare_frozen_evaluations(before, after, protocol(), "c" * 64)
