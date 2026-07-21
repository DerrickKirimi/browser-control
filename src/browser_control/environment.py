"""Backend-independent browser environment contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class BrowserObservation:
    """The text-only state exposed to a browser-control policy."""

    goal: str
    accessibility_tree: str
    last_action_error: str | None = None


@dataclass(frozen=True, slots=True)
class BrowserStep:
    """The complete result of resetting or advancing an environment."""

    observation: BrowserObservation
    reward: float
    terminated: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Whether the current episode must not receive another action."""

        return self.terminated or self.truncated


@runtime_checkable
class BrowserEnvironment(Protocol):
    """Minimum interface required by the trajectory collector."""

    def reset(self, *, seed: int | None = None) -> BrowserStep:
        """Start a new episode."""

    def step(self, action: str) -> BrowserStep:
        """Apply exactly one action to the active episode."""

    def close(self) -> None:
        """Release resources held by the environment."""

