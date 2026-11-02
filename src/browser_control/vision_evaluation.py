"""One-click screenshot evaluation with native rewards and image ablation."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, replace
from pathlib import Path

from browser_control.actions import InvalidActionError
from browser_control.evaluation import wilson_interval
from browser_control.vision_browser import parse_coordinate_click

VISION_TASKS = ("click-color", "click-shape", "click-shades")
IMAGE_CONDITIONS = ("screenshot", "white")


def policy_observation(observation, condition):
    if condition not in IMAGE_CONDITIONS:
        raise ValueError("Unknown image condition")
    if condition == "screenshot":
        return observation
    from PIL import Image

    image = Image.new("RGB", (observation.width, observation.height), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return replace(observation, screenshot_png=output.getvalue())


def _save_image(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Preserve the earlier screenshot: " + path.name)
    else:
        path.write_bytes(data)


def collect_vision_episode(
    environment,
    policy,
    seed,
    image_condition,
    image_directory,
    *,
    generation_seed=None,
    record_logprobs=False,
):
    """One reset, one model call and at most one validated coordinate action."""
    generation_seed = seed * 16 if generation_seed is None else generation_seed
    before = environment.reset(seed=seed)
    if before.done:
        raise ValueError("A visual task must start before termination")
    original = before.observation
    observation = policy_observation(original, image_condition)
    original_name = f"seed-{seed}.png"
    input_name = (
        original_name if image_condition == "screenshot" else f"seed-{seed}-white.png"
    )
    image_directory = Path(image_directory)
    _save_image(image_directory / original_name, original.screenshot_png)
    _save_image(image_directory / input_name, observation.screenshot_png)
    numerical = policy.check_logits(observation)
    if not numerical["finite"]:
        raise RuntimeError("Nonfinite visual policy; no success statistics")
    output = policy.generate(
        observation,
        seed=generation_seed,
        **({"record_logprobs": True} if record_logprobs else {}),
    )
    if (
        not 0 < output.completion_tokens <= 128
        or output.completion_tokens != len(output.completion_token_ids)
        or output.prompt_tokens != len(output.prompt_token_ids)
        or output.generation_logits_checks != output.completion_tokens
        or not math.isfinite(output.generation_seconds)
        or output.generation_seconds < 0
    ):
        raise ValueError("Invalid raw generation evidence")
    row = {
        "task_seed": seed,
        "generation_seed": generation_seed,
        "goal": observation.goal,
        "image_condition": image_condition,
        "screenshot_filename": original_name,
        "input_image_filename": input_name,
        "screenshot_sha256": hashlib.sha256(original.screenshot_png).hexdigest(),
        "input_image_sha256": hashlib.sha256(observation.screenshot_png).hexdigest(),
        "width": observation.width,
        "height": observation.height,
        "numerical_check": numerical,
        "generation": asdict(output),
        "valid_action": False,
        "environment_steps": 0,
        "native_reward": 0.0,
        "done": False,
        "success": False,
    }
    try:
        action = parse_coordinate_click(output.text)
    except InvalidActionError as error:
        row["error"] = str(error)
        return row
    row.update(
        valid_action=True,
        normalized_coordinates={"x": action.x, "y": action.y},
        action=action.render(observation.width, observation.height),
    )
    after = environment.step(action)
    if not math.isfinite(after.reward):
        raise RuntimeError("Nonfinite native environment reward")
    row.update(
        environment_steps=1,
        native_reward=after.reward,
        done=after.done,
        terminated=after.terminated,
        truncated=after.truncated,
        success=bool(after.done and after.reward > 0),
        action_error=after.observation.last_action_error,
    )
    return row


def episode_outcome(row):
    """Separate a wrong visual decision from output that never reached the page."""
    if row["success"] != bool(row["done"] and row["native_reward"] > 0):
        raise ValueError("Success differs from the native terminal outcome")
    if row["success"]:
        return "success"
    return "wrong-click" if row["valid_action"] else "malformed"


def classify_variation_group(rows):
    """Describe one task's samples; only success against a wrong click is a decision."""
    if not rows or len({row["task_seed"] for row in rows}) != 1:
        raise ValueError("A variation group holds samples of one task")
    outcomes = [episode_outcome(row) for row in rows]
    kinds = set(outcomes)
    return {
        "task_seed": rows[0]["task_seed"],
        "outcomes": outcomes,
        "token_sequences": len(
            {tuple(row["generation"]["completion_token_ids"]) for row in rows}
        ),
        "native_reward_varied": "success" in kinds and len(kinds) > 1,
        "decision_varied": {"success", "wrong-click"} <= kinds,
    }


def summarize_vision_episodes(episodes):
    if not episodes or len({row["task_seed"] for row in episodes}) != len(episodes):
        raise ValueError("Expected nonempty distinct visual task seeds")
    for row in episodes:
        if row["success"] != bool(row["done"] and row["native_reward"] > 0):
            raise ValueError("Success differs from the native terminal outcome")
    successes = sum(row["success"] for row in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "wilson_95": wilson_interval(successes, len(episodes)),
        "valid_calls": sum(row["valid_action"] for row in episodes),
        "action_errors": sum(bool(row.get("action_error")) for row in episodes),
        "mean_environment_steps": sum(row["environment_steps"] for row in episodes)
        / len(episodes),
        "mean_completion_tokens": sum(
            row["generation"]["completion_tokens"] for row in episodes
        )
        / len(episodes),
    }
