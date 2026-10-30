"""Validate continuation of an interrupted frozen evaluation."""

from __future__ import annotations

from typing import Any


def validate_frozen_resume(previous: dict[str, Any], current: dict[str, Any]) -> int:
    """Require unchanged conditions and policy code, and a contiguous seed prefix."""
    if previous.get("status") != "running" or previous.get("split") != current["split"]:
        raise ValueError("resume requires an interrupted report from the same split")
    if previous["manifest"]["config"] != current["manifest"]["config"]:
        raise ValueError("evaluation configuration changed; cannot resume")
    if previous["generation_config"] != current["generation_config"]:
        raise ValueError("generation configuration changed; cannot resume")
    old_files = previous["source_snapshot"]["files_sha256"]
    new_files = current["source_snapshot"]["files_sha256"]
    for name, digest in old_files.items():
        if name.startswith("src/") and new_files.get(name) != digest:
            raise ValueError(f"policy/environment source changed: {name}")
    seeds = current["manifest"]["config"]["task_seeds"]
    retained = [episode["task_seed"] for episode in previous["episodes"]]
    if not retained or retained != seeds[: len(retained)] or len(retained) > len(seeds):
        raise ValueError("retained episodes must be a nonempty contiguous seed prefix")
    return len(retained)
