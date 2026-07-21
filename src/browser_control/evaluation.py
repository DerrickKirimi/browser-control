"""Seeded click-test evaluation independent of model implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Protocol

from browser_control.actions import InvalidActionError, parse_click_test_action
from browser_control.environment import BrowserEnvironment
from browser_control.prompts import build_click_test_messages


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    text: str
    generation_seconds: float
    prompt_tokens: int
    completion_tokens: int


class TextPolicy(Protocol):
    def generate(
        self, messages: list[dict[str, str]], *, seed: int
    ) -> PolicyOutput: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    model_output: str
    parsed_action: str | None
    valid_action: bool
    success: bool
    reward: float
    environment_steps: int
    generation_seconds: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    episodes: tuple[EpisodeResult, ...]
    success_rate: float
    invalid_action_rate: float
    mean_environment_steps: float
    mean_generation_seconds: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["episodes"] = [asdict(episode) for episode in self.episodes]
        return payload


def evaluate_click_test(
    environment_factory: Callable[[], BrowserEnvironment],
    policy: TextPolicy,
    seeds: Iterable[int],
) -> EvaluationReport:
    """Evaluate one policy action per seeded click-test episode."""

    episode_results: list[EpisodeResult] = []
    environment = environment_factory()
    try:
        for seed in seeds:
            initial = environment.reset(seed=seed)
            output = policy.generate(
                build_click_test_messages(initial.observation), seed=seed
            )
            try:
                parsed = parse_click_test_action(output.text)
            except InvalidActionError as error:
                episode_results.append(
                    EpisodeResult(
                        seed=seed,
                        model_output=output.text,
                        parsed_action=None,
                        valid_action=False,
                        success=False,
                        reward=0.0,
                        environment_steps=0,
                        generation_seconds=output.generation_seconds,
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                        error=str(error),
                    )
                )
                continue

            action = parsed.render()
            step = environment.step(action)
            episode_results.append(
                EpisodeResult(
                    seed=seed,
                    model_output=output.text,
                    parsed_action=action,
                    valid_action=True,
                    success=step.done and step.reward > 0,
                    reward=step.reward,
                    environment_steps=1,
                    generation_seconds=output.generation_seconds,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                )
            )
    finally:
        environment.close()

    if not episode_results:
        raise ValueError("evaluation requires at least one episode seed")
    return EvaluationReport(
        episodes=tuple(episode_results),
        success_rate=mean(result.success for result in episode_results),
        invalid_action_rate=mean(
            not result.valid_action for result in episode_results
        ),
        mean_environment_steps=mean(
            result.environment_steps for result in episode_results
        ),
        mean_generation_seconds=mean(
            result.generation_seconds for result in episode_results
        ),
    )

