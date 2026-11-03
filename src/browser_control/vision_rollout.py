"""Sample screenshot groups for native-reward GRPO, keeping the exact IDs that were generated."""

from __future__ import annotations

import io
import json
from pathlib import Path

from browser_control.vision_evaluation import collect_vision_episode


def fit_generation_seed(task_seed: int, member: int) -> int:
    """Distinct from evaluation seeds, and deterministic per sample."""
    return 2_000_000 + task_seed * 64 + member * 16


class VisionGroupCollector:
    """One action per episode, so every generated token is policy output.

    Malformed output stays in its group with zero native reward. Nothing repairs it.
    """

    def __init__(self, environment, policy, image_directory):
        self.environment = environment
        self.policy = policy
        self.image_directory = Path(image_directory)
        self.episodes = []
        self.groups = []

    def __call__(self, inputs, trainer):
        from PIL import Image

        seeds = [json.loads(item["prompt"])["task_seed"] for item in inputs]
        if len(set(seeds)) != 1:
            raise ValueError("Each collection must hold one task's group")
        was_training = self.policy.model.training
        self.policy.model.eval()
        rows = []
        try:
            for member, seed in enumerate(seeds):
                episode = collect_vision_episode(
                    self.environment,
                    self.policy,
                    seed,
                    "screenshot",
                    self.image_directory,
                    generation_seed=fit_generation_seed(seed, member),
                    record_logprobs=True,
                )
                generation = episode["generation"]
                image = Image.open(
                    io.BytesIO(
                        (self.image_directory / episode["input_image_filename"]).read_bytes()
                    )
                ).convert("RGB")
                rows.append(
                    {
                        "prompt_ids": list(generation["prompt_token_ids"]),
                        "completion_ids": list(generation["completion_token_ids"]),
                        "env_mask": [1] * len(generation["completion_token_ids"]),
                        "logprobs": list(generation["sampling_logprobs"]),
                        "native_success": bool(episode["success"]),
                        "images": [image],
                        "processing_messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": episode["goal"]},
                                ],
                            }
                        ],
                    }
                )
                self.episodes.append(episode)
        finally:
            if was_training:
                self.policy.model.train()
        rewards = [row["native_success"] for row in rows]
        self.groups.append(
            {
                "task_seed": seeds[0],
                "native_rewards": [float(value) for value in rewards],
                "reward_variation": len(set(rewards)) == 2,
            }
        )
        return rows
