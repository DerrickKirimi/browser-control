from __future__ import annotations

from typing import Any

import pytest

from browser_control.environment import BrowserEnvironment
from browser_control.local_browsergym import LocalBrowserGymEnvironment


class StubGymEnvironment:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.seed: int | None = None
        self.closed = False

    def reset(self, *, seed: int | None = None):
        self.seed = seed
        return (
            {
                "goal_object": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Click the button."}],
                    }
                ],
                "axtree_txt": "[13] button 'Click Me!'",
                "last_action_error": False,
            },
            {"reset": True},
        )

    def step(self, action: str):
        self.actions.append(action)
        success = action == "click('13')"
        observation: dict[str, Any] = {
            "goal": "Click the button.",
            "axtree_txt": "[13] button 'Click Me!'",
            "last_action_error": not success,
        }
        return observation, float(success), True, False, {"success": success}

    def close(self) -> None:
        self.closed = True


def test_adapter_satisfies_environment_protocol() -> None:
    environment = LocalBrowserGymEnvironment(StubGymEnvironment())

    assert isinstance(environment, BrowserEnvironment)


def test_adapter_normalizes_reset_and_step() -> None:
    gym_environment = StubGymEnvironment()
    environment = LocalBrowserGymEnvironment(gym_environment)

    initial = environment.reset(seed=23)
    result = environment.step("click('13')")

    assert gym_environment.seed == 23
    assert initial.observation.goal == "Click the button."
    assert initial.observation.accessibility_tree == "[13] button 'Click Me!'"
    assert initial.info == {"reset": True}
    assert result.reward == 1.0
    assert result.terminated
    assert not result.truncated
    assert result.info == {"success": True}
    assert gym_environment.actions == ["click('13')"]


def test_adapter_surfaces_browsergym_action_error() -> None:
    environment = LocalBrowserGymEnvironment(StubGymEnvironment())
    environment.reset()

    result = environment.step("noop()")

    assert result.reward == 0.0
    assert result.observation.last_action_error == (
        "BrowserGym reported an action error"
    )


def test_adapter_enforces_episode_lifecycle_and_closes_backend() -> None:
    gym_environment = StubGymEnvironment()
    environment = LocalBrowserGymEnvironment(gym_environment)

    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        environment.step("noop()")

    environment.reset()
    environment.step("noop()")
    with pytest.raises(RuntimeError, match="after the episode is done"):
        environment.step("noop()")

    environment.close()
    assert gym_environment.closed

