"""Locked policy comparison on the reserved book-flight cohort."""

from __future__ import annotations

import re
from dataclasses import asdict

from browser_control.book_flight_comparison import (
    SETTINGS,
    _compare_evaluations,
    _config,
    _episodes,
)
from browser_control.book_flight_diagnostic import SAMPLING_128
from browser_control.evaluation import wilson_interval
from browser_control.lfm3 import MODEL_REVISION

RESERVED_SEEDS = range(10100, 10300)
FROZEN_SETTINGS = {
    "model_revision": MODEL_REVISION,
    "decoder": asdict(SAMPLING_128),
    "generation_seed": "task_seed * 16 + turn",
    "attention": "sdpa",
    "max_prompt_tokens": 16384,
    "prefill_chunk_size": 512,
    "max_steps": 15,
    "enable_thinking": False,
}
ARMS_BY_SCHEMA = {
    "book-flight-frozen-v1": {"sft", "sft_grpo"},
    # Declared when no GRPO policy exists to compare; it supports no GRPO claim.
    "book-flight-frozen-sft-only-v1": {"sft"},
}


def validate_frozen_protocol(protocol):
    """Require every policy hash and all settings before any arm is run."""
    schema = protocol.get("schema")
    if schema not in ARMS_BY_SCHEMA:
        raise ValueError("Expected a book-flight-frozen protocol")
    seeds = protocol.get("task_seeds")
    if seeds != list(RESERVED_SEEDS) or any(type(seed) is not int for seed in seeds):
        raise ValueError("Expected the entire reserved cohort10100–10299")
    if protocol.get("settings") != FROZEN_SETTINGS:
        raise ValueError("Frozen evaluation settings differ")
    arms = protocol.get("arms", {})
    if set(arms) != ARMS_BY_SCHEMA[schema] or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in arms.values()
    ):
        raise ValueError(f"Lock adapter SHA256 hashes for {sorted(ARMS_BY_SCHEMA[schema])}")


def validate_frozen_arm(protocol, arm, adapter_sha256):
    validate_frozen_protocol(protocol)
    if arm not in protocol["arms"] or protocol["arms"][arm] != adapter_sha256:
        raise ValueError("Adapter differs from the locked policy arm")


def _check_frozen_report(report, protocol, protocol_sha256, arm):
    config = report["manifest"]["config"]
    validate_frozen_arm(protocol, arm, config["adapter_sha256"])
    if (
        config.get("frozen_arm") != arm
        or config.get("frozen_protocol_sha256") != protocol_sha256
        or {key: config.get(key) for key in SETTINGS} != protocol["settings"]
    ):
        raise ValueError("Evaluation differs from the frozen protocol")


def summarize_frozen_arm(report, protocol, protocol_sha256, arm="sft"):
    """Recompute one arm's native successes; an estimate, not a comparison."""
    validate_frozen_protocol(protocol)
    _check_frozen_report(report, protocol, protocol_sha256, arm)
    rows = _episodes(report, _config(report, None), RESERVED_SEEDS)
    successes = sum(row["success"] for row in rows)
    return {
        "scope": "reserved single-policy estimate; not a comparison with any other policy",
        "frozen_protocol_sha256": protocol_sha256,
        "arm": arm,
        "episodes": len(rows),
        "successes": successes,
        "wilson_95": wilson_interval(successes, len(rows)),
        "failed_task_seeds": [row["task_seed"] for row in rows if not row["success"]],
    }


def compare_frozen_evaluations(baseline, candidate, protocol, protocol_sha256):
    """Recompute native pairs; marginal Wilson intervals are not delta intervals."""
    validate_frozen_protocol(protocol)
    if protocol["schema"] != "book-flight-frozen-v1":
        raise ValueError("This protocol locks one arm; there is nothing to compare")
    for report, arm in ((baseline, "sft"), (candidate, "sft_grpo")):
        _check_frozen_report(report, protocol, protocol_sha256, arm)
    result = _compare_evaluations(baseline, candidate, seeds=RESERVED_SEEDS)
    samples = result["episodes"]
    return {
        **result,
        "scope": "reserved paired policy comparison; marginal intervals do not test the paired difference",
        "frozen_protocol_sha256": protocol_sha256,
        "success_rate_difference": result["net_wins"] / samples,
        "baseline_wilson_95": wilson_interval(result["baseline_successes"], samples),
        "candidate_wilson_95": wilson_interval(result["candidate_successes"], samples),
    }
