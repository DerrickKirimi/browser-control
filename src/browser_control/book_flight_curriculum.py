"""Explicit training-only skill selection and repeat budgets for expert SFT."""

from __future__ import annotations

import copy
import random

from browser_control.book_flight_sft import TRAINING_SEEDS


def plan_examples(
    examples: list[dict], *, skills=None, passes=1, seed=23, replay_per_skill=0
) -> list[dict]:
    if passes < 1:
        raise ValueError("Training passes must be positive")
    if replay_per_skill < 0 or (replay_per_skill and not skills):
        raise ValueError("Replay requires focus skills and a nonnegative budget")
    if any(example["task_seed"] not in TRAINING_SEEDS for example in examples):
        raise ValueError("Curriculum accepts training seeds only")
    selected = list(examples)
    if skills is not None:
        if not skills or set(skills) - {e["skill"] for e in examples}:
            raise ValueError("Requested skills must exist in the expert corpus")
        selected = [e for e in examples if e["skill"] in skills]
    planned = []
    for index in range(passes):
        rng = random.Random(seed + index)
        current = [
            {
                **copy.deepcopy(example),
                "pass_index": index,
                "curriculum_role": "focus" if skills else "primary",
            }
            for example in selected
        ]
        if replay_per_skill:
            for skill in sorted({e["skill"] for e in examples} - set(skills)):
                pool = [e for e in examples if e["skill"] == skill]
                if len(pool) < replay_per_skill:
                    raise ValueError(
                        "Replay budget exceeds available examples for a skill"
                    )
                current.extend(
                    {
                        **copy.deepcopy(e),
                        "pass_index": index,
                        "curriculum_role": "replay",
                    }
                    for e in rng.sample(pool, replay_per_skill)
                )
        rng.shuffle(current)
        planned.extend(current)
    return planned


def validate_initial_adapter(
    config: dict, expected: dict, *, revision: str, model_id: str
):
    if config.get("base_model_name_or_path") != model_id:
        raise ValueError("Initial adapter names a different base model")
    if config.get("revision") not in (None, revision):
        raise ValueError("Initial adapter names a different pinned revision")
    for key in ("r", "lora_alpha", "lora_dropout", "task_type"):
        if config.get(key) != expected[key]:
            raise ValueError(f"Initial adapter differs in {key}")
    if not isinstance(config.get("target_modules"), list) or set(
        config["target_modules"]
    ) != set(expected["target_modules"]):
        raise ValueError("Initial adapter uses different LoRA targets")
    if config.get("bias", "none") != "none":
        raise ValueError("Initial adapter must not train base biases")
    for key in (
        "modules_to_save",
        "rank_pattern",
        "alpha_pattern",
        "layer_replication",
        "layers_to_transform",
        "use_dora",
        "use_rslora",
        "lora_bias",
        "fan_in_fan_out",
    ):
        if config.get(key):
            raise ValueError(f"Unsupported initial adapter override: {key}")
