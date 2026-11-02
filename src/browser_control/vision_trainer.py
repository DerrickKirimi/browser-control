"""Dynamic image transfer and raw-token guards for native-reward TRL1.16 rollouts."""

from __future__ import annotations

import hashlib
import math
import statistics

from browser_control.book_flight_rollout import (
    native_success_reward,
    validate_trainer_token_rows,
)
from browser_control.lfm3_trainer import low_scale_grpo_trainer_class


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}:{list(value.shape)}".encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class StagedVisionRollout:
    """Collect before TRL extracts images, then hand back the unchanged sampled IDs.

    The collector returns raw token rows, native_success, images and
    processing_messages. Those messages prepare image tensors only; the model
    trains on the collector's raw prompt/completion IDs, including tool images.
    """

    def __init__(self, collector):
        self.collector = collector
        self.episodes = []
        self._staged = None

    def prepare_inputs(self, inputs, trainer):
        import torch

        if self._staged is not None:
            raise ValueError("Previous visual rollout was not consumed")
        rows = self.collector(inputs, trainer)
        size = trainer.num_generations
        if not rows or len(rows) != len(inputs) or len(rows) % size:
            raise ValueError("Expected complete aligned visual generation groups")
        for row in rows:
            if (
                not row["prompt_ids"]
                or not row["completion_ids"]
                or len(row["completion_ids"]) != len(row["logprobs"])
                or len(row["completion_ids"]) != len(row["env_mask"])
                or type(row["native_success"]) is not bool
                or not row["images"]
                or not row["processing_messages"]
                or any(
                    mask not in (0, 1) or not math.isfinite(p) or (not mask and p != 0)
                    for mask, p in zip(row["env_mask"], row["logprobs"])
                )
            ):
                raise ValueError("Invalid visual token/image/native row")
            image_inputs = trainer.processing_class.image_processor(
                images=row["images"], return_tensors="pt"
            )
            row["image_binding"] = {
                "num_images": len(row["images"]),
                "grid_sha256": tensor_sha(image_inputs["image_grid_thw"]),
                "pixel_sha256": {
                    str(dtype): tensor_sha(image_inputs["pixel_values"].to(dtype=dtype))
                    for dtype in (torch.float32, torch.float16)
                },
            }
        for start in range(0, len(rows), size):
            group = rows[start : start + size]
            rewards = [float(row["native_success"]) for row in group]
            mean, std = statistics.mean(rewards), statistics.stdev(rewards)
            for row, reward in zip(group, rewards):
                row["expected_advantage"] = (reward - mean) / (std + 1e-4)
        self.episodes.extend(rows)
        self._staged = rows
        return [
            {**original, "prompt": row["processing_messages"], "images": row["images"]}
            for original, row in zip(inputs, rows)
        ]

    def __call__(self, prompts, trainer):
        if self._staged is None or len(prompts) != len(self._staged):
            raise ValueError("Visual rollout must be prepared exactly once")
        rows, self._staged = self._staged, None
        return {
            field: [row[field] for row in rows]
            for field in (
                "prompt_ids",
                "completion_ids",
                "logprobs",
                "env_mask",
                "native_success",
            )
        }


def validate_visual_trainer_rows(inputs, episodes):
    """Bind actual microbatch pixels, grids and advantages to the sampled rows."""
    validate_trainer_token_rows(inputs, episodes)
    grid, pixels = inputs["image_grid_thw"], inputs["pixel_values"]
    counts = inputs["num_images"]
    if (
        grid.ndim != 2
        or grid.shape[1] != 3
        or pixels.ndim != 2
        or sum(counts) != grid.shape[0]
        or int(grid.prod(-1).sum()) != pixels.shape[0]
    ):
        raise ValueError("Visual grid/patch counts differ")
    checks = []
    image_offset = patch_offset = 0
    for index, count in enumerate(counts):
        row_grid = grid[image_offset : image_offset + count]
        patches = int(row_grid.prod(-1).sum())
        row_pixels = pixels[patch_offset : patch_offset + patches]
        binding = {
            "num_images": count,
            "grid_sha256": tensor_sha(row_grid),
            "pixel_sha256": tensor_sha(row_pixels),
        }
        prompt = inputs["prompt_ids"][index][
            inputs["prompt_mask"][index].bool()
        ].tolist()
        completion = inputs["completion_ids"][index][
            inputs["completion_mask"][index].bool()
        ].tolist()
        candidates = [
            row
            for row in episodes
            if row["prompt_ids"] == prompt and row["completion_ids"] == completion
        ]
        matched = [
            row
            for row in candidates
            if row["image_binding"]["num_images"] == count
            and row["image_binding"]["grid_sha256"] == binding["grid_sha256"]
            and row["image_binding"]["pixel_sha256"].get(str(row_pixels.dtype))
            == binding["pixel_sha256"]
            and math.isclose(
                row["expected_advantage"],
                float(inputs["advantages"][index]),
                abs_tol=1e-6,
            )
        ]
        if not matched:
            raise ValueError(
                "Trainer images or native advantage differ from the sampled row"
            )
        checks.append(binding)
        image_offset += count
        patch_offset += patches
    if len(counts) != len(inputs["prompt_ids"]):
        raise ValueError("Image counts do not cover the microbatch")
    return checks


def native_vision_grpo_trainer_class():
    """Construct the pinned trainer with staged images and mandatory live guards."""
    base = low_scale_grpo_trainer_class()

    class NativeVisionGRPOTrainer(base):
        def __init__(self, *args, rollout_func, **kwargs):
            if not isinstance(rollout_func, StagedVisionRollout):
                raise ValueError("Native vision training requires staged raw rollouts")
            self.vision_rollout = rollout_func
            self.vision_binding_checks = []
            self.vision_loss_checks = []
            super().__init__(*args, rollout_func=rollout_func, **kwargs)
            if (
                self.args.scale_rewards != "group"
                or self.args.beta != 0
                or self.args.loss_type != "grpo"
                or self.args.num_iterations != 1
                or self.reward_funcs != [native_success_reward]
                or self.reward_weights.tolist() != [1.0]
            ):
                raise ValueError(
                    "Image/native guards require one-iteration GRPO, unit native rewards, group scaling and beta0"
                )

        def _generate_and_score_completions(self, inputs):
            prepared = self.vision_rollout.prepare_inputs(inputs, self)
            return super()._generate_and_score_completions(prepared)

        def _compute_loss(self, model, inputs):
            self.vision_binding_checks.extend(
                validate_visual_trainer_rows(inputs, self.vision_rollout.episodes)
            )
            self._vision_active = inputs["completion_mask"] * inputs["tool_mask"]
            self.vision_loss_checks.append(
                {
                    "raw_ids_preserved": True,
                    "images_preserved": True,
                    "nonzero_advantage": bool((inputs["advantages"] != 0).any()),
                }
            )
            return super()._compute_loss(model, inputs)

        def _get_per_token_logps_and_entropies(self, *args, **kwargs):
            import torch

            result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
            if result[0].requires_grad:
                active = self._vision_active
                check = self.vision_loss_checks[-1]

                def verify_gradient(gradient):
                    if (
                        not torch.isfinite(gradient).all()
                        or torch.any(gradient[active == 0] != 0)
                        or bool(torch.any(gradient[active == 1] != 0))
                        != check["nonzero_advantage"]
                    ):
                        raise ValueError(
                            "Visual policy gradient differs from the generated/native mask"
                        )
                    check["external_gradients_zero"] = True

                result[0].register_hook(verify_gradient)
            return result

    return NativeVisionGRPOTrainer
