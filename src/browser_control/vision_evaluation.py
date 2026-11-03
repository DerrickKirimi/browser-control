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
# Tasks that need more than one click; evaluated only with --max-steps above 1.
MULTI_STEP_TASKS = ("click-button-sequence",)
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


def history_goal(goal, clicks):
    """The goal plus earlier clicks as text; unchanged on the first turn."""
    if not clicks:
        return goal
    listed = ", ".join(f"({x}, {y})" for x, y in clicks)
    return f"{goal}\nEarlier clicks (x, y on 0–1000): {listed}"


def collect_vision_sequence(
    environment,
    policy,
    seed,
    image_condition,
    image_directory,
    *,
    max_steps,
    generation_seed=None,
    record_logprobs=False,
):
    """Up to max_steps turns of screenshot, one model call and one validated click.

    Each turn sees a fresh screenshot and the earlier clicks as text. An invalid call ends
    the episode unrepaired; success is the native terminal reward, as in one-click runs.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    base_seed = seed * 16 if generation_seed is None else generation_seed
    step = environment.reset(seed=seed)
    if step.done:
        raise ValueError("A visual task must start before termination")
    image_directory = Path(image_directory)
    goal = step.observation.goal
    turns, clicks = [], []
    for turn in range(max_steps):
        original = replace(step.observation, goal=history_goal(goal, clicks))
        observation = policy_observation(original, image_condition)
        # Later turns depend on earlier clicks, so samples of one task get their own files.
        suffix = "" if turn == 0 else f"-g{base_seed}-turn-{turn}"
        original_name = f"seed-{seed}{suffix}.png"
        input_name = original_name if image_condition == "screenshot" else f"seed-{seed}{suffix}-white.png"
        _save_image(image_directory / original_name, original.screenshot_png)
        _save_image(image_directory / input_name, observation.screenshot_png)
        numerical = policy.check_logits(observation)
        if not numerical["finite"]:
            raise RuntimeError("Nonfinite visual policy; no success statistics")
        output = policy.generate(
            observation,
            seed=base_seed + turn,
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
        record = {
            "turn": turn,
            "generation_seed": base_seed + turn,
            "prompt_text": observation.goal,
            "screenshot_filename": original_name,
            "input_image_filename": input_name,
            "screenshot_sha256": hashlib.sha256(original.screenshot_png).hexdigest(),
            "input_image_sha256": hashlib.sha256(observation.screenshot_png).hexdigest(),
            "numerical_check": numerical,
            "generation": asdict(output),
            "valid_action": False,
        }
        turns.append(record)
        try:
            action = parse_coordinate_click(output.text)
        except InvalidActionError as error:
            record["error"] = str(error)
            break
        step = environment.step(action)
        if not math.isfinite(step.reward):
            raise RuntimeError("Nonfinite native environment reward")
        clicks.append((action.x, action.y))
        record.update(
            valid_action=True,
            normalized_coordinates={"x": action.x, "y": action.y},
            action=action.render(observation.width, observation.height),
            native_reward=step.reward,
            done=step.done,
            action_error=step.observation.last_action_error,
        )
        if step.done:
            break
    last = turns[-1]
    done = bool(last.get("done"))
    reward = float(last.get("native_reward", 0.0))
    return {
        "task_seed": seed,
        "goal": goal,
        "image_condition": image_condition,
        "max_steps": max_steps,
        "turns": turns,
        "environment_steps": len(clicks),
        "valid_action": all(t["valid_action"] for t in turns),
        "native_reward": reward,
        "done": done,
        "success": bool(done and reward > 0),
        "stop_reason": "done" if done else ("invalid-call" if not last["valid_action"] else "step-limit"),
        "action_error": next((t["action_error"] for t in turns if t.get("action_error")), None),
        # Concatenated per-turn evidence, so summaries and variation groups work unchanged.
        "generation": {
            "completion_tokens": sum(t["generation"]["completion_tokens"] for t in turns),
            "completion_token_ids": [i for t in turns for i in (*t["generation"]["completion_token_ids"], -1)],
        },
    }


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


RESERVED_SEEDS = tuple(range(21000, 21100))


def check_reserved_request(protocol, seeds, adapter_sha256, decoder, image_condition):
    """Allow reserved seeds only for a cell and policy that a locked protocol names."""
    if protocol.get("status") != "locked":
        raise ValueError("Reserved evaluation needs a locked protocol")
    declared = list(protocol["seeds"])
    if (
        not declared
        or len(set(declared)) != len(declared)
        or any(seed not in RESERVED_SEEDS for seed in declared)
    ):
        raise ValueError("Protocol seeds must come from the reserved range")
    if list(seeds) != declared:
        raise ValueError("Seeds must equal the protocol's seeds, in order")
    if adapter_sha256 not in protocol["policies"].values():
        raise ValueError("Policy is not named by the protocol")
    if [decoder, image_condition] not in [list(cell) for cell in protocol["cells"]]:
        raise ValueError("Cell is not named by the protocol")
