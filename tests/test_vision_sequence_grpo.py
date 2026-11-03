"""Multi-turn group-relative updates: rewards come only from the page, and every turn shares its episode's advantage."""

import pytest

from browser_control.lfm3_vision import VisionGeneration
from browser_control.vision_browser import VisionObservation, VisionStep
from browser_control.vision_sequence_grpo import (
    group_advantages,
    sample_episode,
    training_turns,
)


def click(x, y):
    return f"<tool_call><function=click><parameter=x>{x}</parameter><parameter=y>{y}</parameter></function></tool_call>"


class Environment:
    def __init__(self, clicks_to_win=2, reward=1.0):
        self.clicks_to_win, self.reward = clicks_to_win, reward
        self.observation = VisionObservation("Click button ONE, then click button TWO.", b"png", 10, 10)

    def reset(self, *, seed):
        self.actions = []
        return VisionStep(self.observation, 0.0, False, False)

    def step(self, action):
        self.actions.append(action)
        done = len(self.actions) >= self.clicks_to_win
        return VisionStep(self.observation, self.reward if done else 0.0, done, False)


class Policy:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def generate(self, observation, *, seed):
        self.calls.append((observation.goal, seed))
        text = self.texts[len(self.calls) - 1]
        return VisionGeneration(text, 0.1, 3, 2, (1, 2, 3), (7, 8), ((1, 2, 2),), 2)


def test_episode_uses_terminal_reward_history_and_turn_seeds():
    policy = Policy([click(100, 200), click(700, 300)])
    episode = sample_episode(Environment(), policy, 120, 5000, max_steps=4)
    assert episode.success and episode.reward == 1.0 and len(episode.turns) == 2
    assert [seed for _, seed in policy.calls] == [5000, 5001]
    assert policy.calls[1][0].endswith("Earlier clicks (x, y on 0–1000): (100, 200)")
    assert all(turn.valid for turn in episode.turns)


def test_invalid_call_ends_the_episode_with_zero_reward():
    episode = sample_episode(Environment(), Policy([click(1, 2), "click(5)"]), 120, 0, max_steps=4)
    assert not episode.success and episode.reward == 0.0
    assert [turn.valid for turn in episode.turns] == [True, False]


def test_advantages_and_turns_carry_signal_only_when_rewards_differ():
    assert group_advantages([1.0, 0.0, 0.0, 1.0]) == [0.5, -0.5, -0.5, 0.5]
    assert group_advantages([0.0, 0.0]) == [0.0, 0.0]
    with pytest.raises(ValueError):
        group_advantages([])
    win = sample_episode(Environment(), Policy([click(1, 1), click(2, 2)]), 1, 0, max_steps=4)
    lose = sample_episode(Environment(), Policy(["no call"]), 1, 0, max_steps=4)
    turns, tokens = training_turns([win, lose], group_advantages([win.reward, lose.reward]))
    assert tokens == 6 and [advantage for _, advantage in turns] == [0.5, 0.5, -0.5]
    flat, _ = training_turns([win, win], group_advantages([1.0, 1.0]))
    assert flat == []
