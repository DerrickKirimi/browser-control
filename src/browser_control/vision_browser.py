"""Screenshot-only observations and bounded coordinate actions for visual tasks."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from browser_control.actions import InvalidActionError
from browser_control.local_browsergym import LocalBrowserGymEnvironment


@dataclass(frozen=True, slots=True)
class VisionObservation:
    """Pixels, task instruction and execution feedback; no DOM or element IDs."""

    goal: str
    screenshot_png: bytes
    width: int
    height: int
    last_action_error: str | None = None


@dataclass(frozen=True, slots=True)
class VisionStep:
    observation: VisionObservation
    reward: float
    terminated: bool
    truncated: bool

    @property
    def done(self):
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class CoordinateClick:
    x: int
    y: int

    def __post_init__(self):
        if any(
            type(value) is not int or not 0 <= value <= 1000
            for value in (self.x, self.y)
        ):
            raise InvalidActionError("Coordinates must be integers in0..1000")

    def render(self, width: int, height: int) -> str:
        """Render screenshot pixels; BrowserGym applies its own page scaling."""
        if any(type(value) is not int or value <= 0 for value in (width, height)):
            raise ValueError("Screenshot dimensions must be positive integers")
        x = round(self.x * (width - 1) / 1000)
        y = round(self.y * (height - 1) / 1000)
        return f"mouse_click({x}, {y})"


def parse_coordinate_click(output: str) -> CoordinateClick:
    """Accept one XML click with exactly x/y, without repairing model output."""
    call = re.fullmatch(
        r"\s*<tool_call>\s*<function=click>\s*(.*?)\s*</function>\s*</tool_call>\s*",
        output,
        flags=re.DOTALL,
    )
    if call is None:
        raise InvalidActionError("Expected exactly one coordinate click tool call")
    parameters = {}
    remaining = call[1].strip()
    while remaining:
        value = re.match(
            r"<parameter=(x|y)>\s*([0-9]{1,4})\s*</parameter>", remaining
        )
        if value is None or value[1] in parameters:
            raise InvalidActionError("Malformed or duplicate coordinate parameter")
        parameters[value[1]] = int(value[2])
        remaining = remaining[value.end() :].strip()
    if set(parameters) != {"x", "y"}:
        raise InvalidActionError("Coordinate click requires exactly x and y")
    return CoordinateClick(**parameters)


class _ScreenshotCapture(LocalBrowserGymEnvironment):
    """Retain pixels at the same reset/step that produces the normalized state."""

    def _normalize(self, *, observation, **kwargs):
        import numpy as np
        from PIL import Image

        pixels = observation.get("screenshot")
        if (
            not isinstance(pixels, np.ndarray)
            or pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[2] != 3
            or min(pixels.shape[:2]) <= 0
        ):
            raise ValueError("Expected an RGB uint8 BrowserGym screenshot")
        image = Image.fromarray(pixels)
        png = io.BytesIO()
        image.save(png, format="PNG")
        self.image_png = png.getvalue()
        self.image_size = image.size
        return super()._normalize(observation=observation, **kwargs)


class VisionBrowserGymEnvironment:
    """Expose only screenshot observations from a coordinate-enabled backend."""

    def __init__(self, gym_environment):
        self._capture = _ScreenshotCapture(gym_environment)

    @classmethod
    def miniwob(cls, task_name: str, **kwargs):
        environment = cls.__new__(cls)
        environment._capture = _ScreenshotCapture.miniwob(
            task_name, coordinate_actions=True, **kwargs
        )
        return environment

    def _visual(self, step):
        return VisionStep(
            observation=VisionObservation(
                goal=step.observation.goal,
                screenshot_png=self._capture.image_png,
                width=self._capture.image_size[0],
                height=self._capture.image_size[1],
                last_action_error=step.observation.last_action_error,
            ),
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
        )

    def reset(self, *, seed=None):
        return self._visual(self._capture.reset(seed=seed))

    def step(self, action: CoordinateClick):
        if not self._capture._active:
            raise RuntimeError("reset() must be called before step()")
        if not isinstance(action, CoordinateClick):
            raise InvalidActionError("Expected a validated CoordinateClick")
        width, height = self._capture.image_size
        return self._visual(self._capture.step(action.render(width, height)))

    def close(self):
        self._capture.close()
