"""In-process BrowserGym adapter used by local and Colab runs."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from browser_control.environment import BrowserObservation, BrowserStep


def _goal_text(observation: Mapping[str, Any]) -> str:
    goal = observation.get("goal")
    if isinstance(goal, str) and goal:
        return goal

    parts: list[str] = []
    goal_object = observation.get("goal_object")
    if not isinstance(goal_object, list):
        return ""
    for message in goal_object:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return " ".join(parts)


def _accessibility_tree(observation: Mapping[str, Any]) -> str:
    tree = observation.get("axtree_txt")
    if isinstance(tree, str) and tree:
        return tree

    tree_object = observation.get("axtree_object")
    if tree_object is None:
        return ""
    try:
        observation_utils = importlib.import_module("browsergym.utils.obs")
        flatten = observation_utils.flatten_axtree_to_str
    except (AttributeError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "BrowserGym returned axtree_object but its text conversion utility "
            "is unavailable"
        ) from error
    return str(flatten(tree_object))


class LocalBrowserGymEnvironment:
    """Normalize a Gymnasium BrowserGym environment to the project protocol."""

    def __init__(self, gym_environment: Any) -> None:
        self._environment = gym_environment
        self._active = False
        self._done = False

    @classmethod
    def miniwob(
        cls,
        task_name: str = "click-test",
        *,
        headless: bool = True,
        miniwob_url: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
        timeout_ms: float | None = None,
        coordinate_actions: bool = False,
    ) -> LocalBrowserGymEnvironment:
        """Create an in-process MiniWoB environment without Docker or OpenEnv."""

        try:
            importlib.import_module("browsergym.miniwob")
            gymnasium = importlib.import_module("gymnasium")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Install the environment extra with "
                "`uv sync --extra environment`, then install Chromium with "
                "`uv run playwright install chromium`."
            ) from error

        environment_kwargs: dict[str, Any] = {"headless": headless}
        if coordinate_actions:
            action_set = importlib.import_module("browsergym.core.action.highlevel")
            environment_kwargs["action_mapping"] = action_set.HighLevelActionSet(
                subsets=["coord"], multiaction=False, strict=True
            ).to_python_code
        if (viewport_width is None) != (viewport_height is None):
            raise ValueError("viewport width and height must be provided together")
        if viewport_width is not None and viewport_height is not None:
            environment_kwargs["viewport"] = {
                "width": viewport_width,
                "height": viewport_height,
            }
        if timeout_ms is not None:
            environment_kwargs["timeout"] = timeout_ms
        if miniwob_url is not None:
            environment_kwargs["task_kwargs"] = {"base_url": miniwob_url}

        environment = gymnasium.make(
            f"browsergym/miniwob.{task_name}", **environment_kwargs
        )
        return cls(environment)

    def reset(self, *, seed: int | None = None) -> BrowserStep:
        observation, info = self._environment.reset(seed=seed)
        self._active = True
        self._done = False
        return self._normalize(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

    def step(self, action: str) -> BrowserStep:
        if not self._active:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            raise RuntimeError("step() cannot be called after the episode is done")

        observation, reward, terminated, truncated, info = self._environment.step(
            action
        )
        result = self._normalize(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )
        self._done = result.done
        return result

    def close(self) -> None:
        self._environment.close()
        self._active = False
        self._done = True

    @staticmethod
    def _normalize(
        *,
        observation: Mapping[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> BrowserStep:
        raw_error = observation.get("last_action_error", False)
        error = (
            raw_error
            if isinstance(raw_error, str) and raw_error
            else "BrowserGym reported an action error"
            if raw_error
            else None
        )
        return BrowserStep(
            observation=BrowserObservation(
                goal=_goal_text(observation),
                accessibility_tree=_accessibility_tree(observation),
                last_action_error=error,
            ),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=dict(info),
        )
