"""Vision evaluation executes only parsed actions and scores native terminal reward."""

import io
from dataclasses import replace

import pytest

from browser_control.lfm3_vision import VisionGeneration
from browser_control.vision_browser import VisionObservation, VisionStep
from browser_control.vision_evaluation import (
    collect_vision_episode,
    policy_observation,
    summarize_vision_episodes,
)

CLICK = "<tool_call><function=click><parameter=x>500</parameter><parameter=y>500</parameter></function></tool_call>"


def observation(color="red"):
    Image = pytest.importorskip("PIL.Image")
    png = io.BytesIO()
    Image.new("RGB", (20, 10), color).save(png, format="PNG")
    return VisionObservation("Click the red box.", png.getvalue(), 20, 10)


class Environment:
    def __init__(self, reward=1.0, done=True):
        self.observation = observation()
        self.reward, self.done = reward, done
        self.actions = []
        self.resets = []

    def reset(self, *, seed):
        self.resets.append(seed)
        return VisionStep(self.observation, 0.0, False, False)

    def step(self, action):
        self.actions.append(action)
        return VisionStep(self.observation, self.reward, self.done, False)


class Policy:
    def __init__(self, text=CLICK, finite=True):
        self.finite = finite
        self.calls = []
        self.output = VisionGeneration(text, 0.1, 2, 1, (1, 2), (3,), ((1, 2, 2),), 1)

    def check_logits(self, observation):
        return {"finite": self.finite}

    def generate(self, observation, *, seed):
        self.calls.append((observation, seed))
        return self.output


def test_one_model_call_and_one_validated_coordinate_action(tmp_path):
    env, policy = Environment(), Policy()
    row = collect_vision_episode(env, policy, 30, "screenshot", tmp_path)
    assert env.resets == [30] and len(env.actions) == 1
    assert row["action"] == "mouse_click(10, 4)"
    assert row["success"] and row["native_reward"] == 1.0
    assert row["generation_seed"] == 480
    assert policy.calls == [(env.observation, 480)]
    assert (
        tmp_path / row["input_image_filename"]
    ).read_bytes() == env.observation.screenshot_png


def test_invalid_output_is_not_repaired_or_executed(tmp_path):
    env, policy = Environment(), Policy("click(123)")
    row = collect_vision_episode(env, policy, 30, "screenshot", tmp_path)
    assert not row["valid_action"] and not row["success"]
    assert row["environment_steps"] == 0 and env.actions == []


@pytest.mark.parametrize("reward,done", [(0.0, True), (-1.0, True), (1.0, False)])
def test_success_requires_positive_native_terminal_reward(tmp_path, reward, done):
    row = collect_vision_episode(
        Environment(reward, done), Policy(), 30, "screenshot", tmp_path
    )
    assert not row["success"]
    assert row["valid_action"] and row["environment_steps"] == 1


def test_white_control_removes_pixels_and_keeps_task_and_coordinate_frame(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    env, policy = Environment(), Policy()
    original = env.observation
    row = collect_vision_episode(env, policy, 30, "white", tmp_path)
    seen = policy.calls[0][0]
    assert seen.goal == original.goal and (seen.width, seen.height) == (20, 10)
    assert Image.open(io.BytesIO(seen.screenshot_png)).getextrema() == ((255, 255),) * 3
    assert row["input_image_sha256"] != row["screenshot_sha256"]
    assert env.observation == original
    assert (
        tmp_path / row["screenshot_filename"]
    ).read_bytes() == original.screenshot_png


def test_numerical_failure_aborts_instead_of_becoming_a_task_failure(tmp_path):
    env, policy = Environment(), Policy(finite=False)
    with pytest.raises(RuntimeError, match="Nonfinite"):
        collect_vision_episode(env, policy, 30, "screenshot", tmp_path)
    assert not policy.calls and not env.actions


def test_nonfinite_native_reward_aborts(tmp_path):
    with pytest.raises(RuntimeError, match="native environment reward"):
        collect_vision_episode(
            Environment(float("nan")), Policy(), 30, "screenshot", tmp_path
        )


def test_raw_generation_counts_must_match_recorded_token_ids(tmp_path):
    env, policy = Environment(), Policy()
    policy.output = replace(policy.output, completion_tokens=2)
    with pytest.raises(ValueError, match="raw generation evidence"):
        collect_vision_episode(env, policy, 30, "screenshot", tmp_path)
    assert not env.actions


def test_earlier_screenshot_cannot_be_overwritten_with_changed_state(tmp_path):
    env = Environment()
    collect_vision_episode(env, Policy(), 30, "screenshot", tmp_path)
    env.observation = observation("blue")
    with pytest.raises(ValueError, match="Preserve"):
        collect_vision_episode(env, Policy(), 30, "screenshot", tmp_path)


def test_native_summary_and_duplicate_seed_rejection(tmp_path):
    row = collect_vision_episode(Environment(), Policy(), 30, "screenshot", tmp_path)
    result = summarize_vision_episodes([row])
    assert result["successes"] == 1 and result["valid_calls"] == 1
    for rows in ([], [row, row]):
        with pytest.raises(ValueError):
            summarize_vision_episodes(rows)
    with pytest.raises(ValueError, match="native terminal"):
        summarize_vision_episodes([{**row, "success": False}])


def test_unknown_image_condition_is_rejected():
    with pytest.raises(ValueError, match="Unknown image condition"):
        policy_observation(observation(), "DOM")
