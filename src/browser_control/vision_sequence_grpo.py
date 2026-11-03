"""Group-relative policy gradient for multi-turn screenshot episodes.

Each task is sampled several times; every episode is scored only by the page's terminal reward. An
episode's advantage is its reward minus the group mean (no division by the standard deviation), and
every turn's completion tokens in that episode share it. Groups whose rewards are all equal carry no
signal and produce no update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from browser_control.actions import InvalidActionError
from browser_control.vision_browser import parse_coordinate_click
from browser_control.vision_evaluation import history_goal


@dataclass
class Turn:
    observation: object
    completion_token_ids: tuple[int, ...]
    text: str
    valid: bool


@dataclass
class Episode:
    task_seed: int
    generation_seed: int
    turns: list[Turn] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False

    @property
    def success(self) -> bool:
        return self.done and self.reward > 0


def sample_episode(environment, policy, task_seed: int, generation_seed: int, max_steps: int) -> Episode:
    """One sampled episode: fresh screenshot and click history each turn; an invalid call ends it."""
    step = environment.reset(seed=task_seed)
    goal, clicks = step.observation.goal, []
    episode = Episode(task_seed, generation_seed)
    for turn in range(max_steps):
        observation = replace(step.observation, goal=history_goal(goal, clicks))
        output = policy.generate(observation, seed=generation_seed + turn)
        try:
            action = parse_coordinate_click(output.text)
        except InvalidActionError:
            episode.turns.append(Turn(observation, tuple(output.completion_token_ids), output.text, False))
            break
        episode.turns.append(Turn(observation, tuple(output.completion_token_ids), output.text, True))
        step = environment.step(action)
        if not math.isfinite(step.reward):
            raise RuntimeError("Nonfinite native environment reward")
        clicks.append((action.x, action.y))
        episode.reward, episode.done = step.reward, step.done
        if step.done:
            break
    return episode


def group_advantages(rewards: list[float]) -> list[float]:
    """Reward minus the group mean; all zero when the group has no variation."""
    if not rewards:
        raise ValueError("Empty group")
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


def training_turns(episodes: list[Episode], advantages: list[float]) -> tuple[list[tuple[Turn, float]], int]:
    """Turns that carry signal, each with its episode's advantage, and the group's token count.

    The loss divides by all completion tokens in the group, so long and short episodes weigh by tokens.
    """
    if len(episodes) != len(advantages):
        raise ValueError("One advantage per episode")
    tokens = sum(len(turn.completion_token_ids) for episode in episodes for turn in episode.turns)
    turns = [(turn, advantage) for episode, advantage in zip(episodes, advantages) if advantage != 0
             for turn in episode.turns]
    return turns, tokens
