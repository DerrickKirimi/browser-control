"""Bounded multi-turn evaluation with observations kept outside generated actions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from browser_control.actions import InvalidActionError
from browser_control.book_flight import (
    SYSTEM_PROMPT,
    initial_messages,
    observation_text,
    parse_book_flight_action,
)
from browser_control.environment import BrowserEnvironment
from browser_control.lfm3 import PromptTooLongError


def collect_book_flight_episode(
    environment: BrowserEnvironment,
    policy: Any,
    task_seed: int,
    *,
    max_steps: int = 15,
    numerical_check: Callable | None = None,
    parse_output: Callable = parse_book_flight_action,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    if not 1 <= max_steps <= 15:
        raise ValueError("max_steps must be in 1..15")
    state = environment.reset(seed=task_seed)
    messages = initial_messages(state.observation, system_prompt)
    record = {
        "task_seed": task_seed,
        "goal": state.observation.goal,
        "steps": [],
        "success": False,
        "environment_steps": 0,
        "action_errors": 0,
        "stop_reason": "horizon",
        "reward": 0.0,
    }
    for turn in range(max_steps):
        generation_seed = task_seed * 16 + turn
        event = {
            "turn": turn,
            "generation_seed": generation_seed,
            "observation": asdict(state.observation),
        }
        try:
            if turn == 0 and numerical_check is not None:
                record["numerical_check"] = numerical_check(messages)
                if not record["numerical_check"]["finite"]:
                    raise RuntimeError("non-finite initial logits")
            output = policy.generate(messages, seed=generation_seed)
        except PromptTooLongError as error:
            event.update(error=str(error), prompt_tokens=error.tokens)
            record["steps"].append(event)
            record["stop_reason"] = "prompt-budget"
            break
        event.update(asdict(output))
        record["steps"].append(event)
        try:
            action = parse_output(output.text)
        except InvalidActionError as error:
            event.update(valid_action=False, error=str(error))
            record["stop_reason"] = "invalid-action"
            break
        event.update(valid_action=True, action=action.render(), tool=action.message())
        state = environment.step(action.render())
        record["environment_steps"] += 1
        record["reward"] = state.reward
        event.update(
            reward=state.reward,
            done=state.done,
            action_error=state.observation.last_action_error,
        )
        record["action_errors"] += bool(state.observation.last_action_error)
        messages.append(action.message())
        if state.done:
            record["success"] = state.reward > 0
            record["stop_reason"] = (
                "success" if record["success"] else "terminal-failure"
            )
            break
        messages.append(
            {"role": "tool", "content": observation_text(state.observation)}
        )
    record["final_observation"] = asdict(state.observation)
    record["model_turns"] = sum("text" in event for event in record["steps"])
    record["success_after_action_error"] = (
        record["success"] and record["action_errors"] > 0
    )
    # This measures recovery from reported execution errors, not all semantic mistakes.
    return record
