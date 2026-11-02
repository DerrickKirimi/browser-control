"""Resume must preserve the exact conditions and already collected outcomes."""

from copy import deepcopy

import pytest

from browser_control.evaluation_resume import validate_frozen_resume


def reports():
    old = {
        "status": "running",
        "split": "heldout",
        "manifest": {"config": {"task_seeds": [10000, 10001, 10002]}},
        "source_snapshot": {
            "files_sha256": {"src/policy.py": "abc", "scripts/run.py": "old"}
        },
        "generation_config": {"prefill_chunk_size": 512},
        "episodes": [{"task_seed": 10000, "success": False}],
    }
    new = deepcopy(old)
    new["source_snapshot"]["files_sha256"]["scripts/run.py"] = "resume-controller"
    new["episodes"] = []
    return old, new


def test_resume_accepts_controller_changes_and_retains_failed_episodes():
    previous, current = reports()
    assert validate_frozen_resume(previous, current) == 1
    assert previous["episodes"] == [{"task_seed": 10000, "success": False}]


@pytest.mark.parametrize(
    "change", ["policy", "decoder", "config", "gap", "duplicate", "complete"]
)
def test_resume_rejects_configuration_or_seed_drift(change):
    previous, current = reports()
    if change == "policy":
        current["source_snapshot"]["files_sha256"]["src/policy.py"] = "changed"
    elif change == "decoder":
        current["generation_config"]["prefill_chunk_size"] = 1024
    elif change == "config":
        current["manifest"]["config"]["task_seeds"] = [20000, 20001]
    elif change == "gap":
        previous["episodes"] = [{"task_seed": 10001}]
    elif change == "duplicate":
        previous["episodes"] *= 2
    else:
        previous["status"] = "complete"
    with pytest.raises(ValueError):
        validate_frozen_resume(previous, current)
