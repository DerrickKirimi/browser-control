"""Matched native development comparisons, recomputed from episode evidence."""

from __future__ import annotations

from copy import deepcopy

from browser_control.book_flight_diagnostic import COMPARISON_SEEDS

SETTINGS = (
    "model_revision",
    "decoder",
    "generation_seed",
    "attention",
    "max_prompt_tokens",
    "prefill_chunk_size",
    "max_steps",
    "enable_thinking",
)


def _config(report, legacy_settings):
    config = deepcopy(report["manifest"]["config"])
    if isinstance(config.get("decoder"), str):
        if not legacy_settings or config["decoder"] != legacy_settings.get(
            "decoder_name"
        ):
            raise ValueError(
                "Legacy decoder labels require explicitly resolved settings"
            )
        config["decoder"] = deepcopy(legacy_settings["decoder"])
    for key in SETTINGS:
        if key not in config and legacy_settings and key in legacy_settings:
            config[key] = deepcopy(legacy_settings[key])
        if key not in config:
            raise ValueError(f"Missing evaluation setting: {key}")
    return config


def _episodes(report, config):
    if report["status"] != "complete":
        raise ValueError("Only completed evaluations can be compared")
    seeds = list(COMPARISON_SEEDS)
    rows = report["episodes"]
    if config.get("task_seeds") != seeds or [row["task_seed"] for row in rows] != seeds:
        raise ValueError("Expected the complete ordered comparison cohort70–99")
    for row in rows:
        if type(row["success"]) is not bool or not row["steps"]:
            raise ValueError("Expected boolean success and recorded model steps")
        final = row["steps"][-1]
        native = bool(final.get("done") and final.get("reward", 0) > 0)
        if native != row["success"]:
            raise ValueError("Success differs from the recorded native terminal reward")
        for index, step in enumerate(row["steps"]):
            if (
                step["turn"] != index
                or step["generation_seed"] != row["task_seed"] * 16 + index
            ):
                raise ValueError(
                    "Generation seeds differ from the frozen evaluation rule"
                )
    return rows


def compare_adapter_evaluations(baseline, candidate, *, legacy_settings=None):
    """Apply the declared +/-4 reading rule; no generalization/significance claim."""
    before_config = _config(baseline, legacy_settings)
    after_config = _config(candidate, legacy_settings)
    for key in SETTINGS:
        if before_config[key] != after_config[key]:
            raise ValueError(f"Evaluation settings differ: {key}")
    before = _episodes(baseline, before_config)
    after = _episodes(candidate, after_config)
    paired = []
    for first, second in zip(before, after, strict=True):
        if (
            first["goal"] != second["goal"]
            or first["steps"][0]["observation"] != second["steps"][0]["observation"]
        ):
            raise ValueError(f"Initial task state differs at seed{first['task_seed']}")
        paired.append(
            {
                "task_seed": first["task_seed"],
                "baseline_success": first["success"],
                "candidate_success": second["success"],
                "identical_action_path": [s.get("action") for s in first["steps"]]
                == [s.get("action") for s in second["steps"]],
            }
        )
    wins = [
        r["task_seed"]
        for r in paired
        if not r["baseline_success"] and r["candidate_success"]
    ]
    losses = [
        r["task_seed"]
        for r in paired
        if r["baseline_success"] and not r["candidate_success"]
    ]
    net = len(wins) - len(losses)
    return {
        "scope": "matched development comparison; +/-4 declared reading rule, not statistical significance or held-out generalization",
        "episodes": len(paired),
        "baseline_successes": sum(r["baseline_success"] for r in paired),
        "candidate_successes": sum(r["candidate_success"] for r in paired),
        "wins": wins,
        "losses": losses,
        "net_wins": net,
        "identical_action_paths": sum(r["identical_action_path"] for r in paired),
        "reading": "candidate-for-replication"
        if net >= 4
        else "harmful-at-these-settings"
        if net <= -4
        else "no-detectable-effect-at-this-budget",
        "evaluation_settings": {key: before_config[key] for key in SETTINGS},
        "paired": paired,
    }
