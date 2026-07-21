"""Deterministic test doubles for browser-control environments."""

from __future__ import annotations

from dataclasses import dataclass, field

from browser_control.environment import BrowserObservation, BrowserStep


@dataclass(slots=True)
class FakeClickTestEnvironment:
    """A one-button task with explicit lifecycle and reward semantics."""

    button_id: str = "13"
    actions: list[str] = field(default_factory=list, init=False)
    _active: bool = field(default=False, init=False, repr=False)
    _done: bool = field(default=False, init=False, repr=False)

    @property
    def expected_action(self) -> str:
        return f"click('{self.button_id}')"

    def reset(self, *, seed: int | None = None) -> BrowserStep:
        del seed
        self.actions.clear()
        self._active = True
        self._done = False
        return BrowserStep(
            observation=self._observation(),
            reward=0.0,
            terminated=False,
            info={"task": "click-test", "backend": "fake"},
        )

    def step(self, action: str) -> BrowserStep:
        if not self._active:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            raise RuntimeError("step() cannot be called after the episode is done")

        self.actions.append(action)
        succeeded = action == self.expected_action
        self._done = True
        error = None if succeeded else f"invalid action: {action}"
        return BrowserStep(
            observation=self._observation(error=error),
            reward=1.0 if succeeded else 0.0,
            terminated=True,
            info={"task": "click-test", "success": succeeded, "backend": "fake"},
        )

    def close(self) -> None:
        self._active = False

    def _observation(self, *, error: str | None = None) -> BrowserObservation:
        return BrowserObservation(
            goal="Click the button.",
            accessibility_tree=f"[{self.button_id}] button 'Click Me!'",
            last_action_error=error,
        )

