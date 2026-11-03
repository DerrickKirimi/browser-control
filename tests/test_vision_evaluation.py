"""Vision evaluation executes only parsed actions and scores native terminal reward."""

import io
from dataclasses import replace

import pytest

from browser_control.lfm3_vision import VisionGeneration
from browser_control.vision_browser import VisionObservation, VisionStep
from browser_control.vision_evaluation import (
    classify_variation_group,
    collect_vision_episode,
    episode_outcome,
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


def _variation_row(outcome, token=1):
    return {
        "task_seed": 100,
        "success": outcome == "success",
        "done": outcome != "malformed",
        "native_reward": float(outcome == "success"),
        "valid_action": outcome != "malformed",
        "generation": {"completion_token_ids": [token]},
    }


def test_only_success_against_a_wrong_click_is_a_decision():
    decision = classify_variation_group(
        [_variation_row("success"), _variation_row("wrong-click", 2)]
    )
    assert decision["decision_varied"] and decision["native_reward_varied"]
    assert decision["token_sequences"] == 2
    formatting = classify_variation_group(
        [_variation_row("success"), _variation_row("malformed", 2)]
    )
    assert formatting["native_reward_varied"] and not formatting["decision_varied"]
    misses = classify_variation_group(
        [_variation_row("wrong-click"), _variation_row("malformed", 2)]
    )
    assert not misses["native_reward_varied"] and not misses["decision_varied"]


def test_variation_group_refuses_mixed_tasks_and_false_labels():
    other = {**_variation_row("success"), "task_seed": 101}
    with pytest.raises(ValueError, match="one task"):
        classify_variation_group([_variation_row("success"), other])
    with pytest.raises(ValueError, match="native"):
        classify_variation_group([{**_variation_row("malformed"), "success": True}])


def test_requested_rng_and_logprob_recording_reach_the_policy(tmp_path):
    class Policy:
        calls = []

        def check_logits(self, _observation):
            return {"finite": True}

        def generate(self, _observation, **kwargs):
            self.calls.append(kwargs)
            return VisionGeneration(CLICK, 0.1, 2, 1, (1, 2), (3,), ((1, 2, 2),), 1)

    row = collect_vision_episode(
        Environment(),
        Policy(),
        100,
        "screenshot",
        tmp_path,
        generation_seed=77,
        record_logprobs=True,
    )
    assert Policy.calls == [{"seed": 77, "record_logprobs": True}]
    assert row["generation_seed"] == 77


def reserved_protocol():
    return {
        "status": "locked",
        "seeds": [21000, 21001],
        "policies": {"base": None, "grpo": "a" * 64},
        "cells": [["sampling", "screenshot"], ["greedy", "screenshot"]],
    }


def test_reserved_request_accepts_only_declared_cells_and_policies():
    from browser_control.vision_evaluation import check_reserved_request

    protocol = reserved_protocol()
    check_reserved_request(protocol, [21000, 21001], None, "sampling", "screenshot")
    check_reserved_request(protocol, [21000, 21001], "a" * 64, "greedy", "screenshot")
    bad = [
        ({**protocol, "status": "draft"}, [21000, 21001], None, "sampling", "screenshot"),
        ({**protocol, "seeds": [60, 61]}, [60, 61], None, "sampling", "screenshot"),
        ({**protocol, "seeds": [21000, 21000]}, [21000, 21000], None, "sampling", "screenshot"),
        (protocol, [21001, 21000], None, "sampling", "screenshot"),
        (protocol, [21000], None, "sampling", "screenshot"),
        (protocol, [21000, 21001], "b" * 64, "sampling", "screenshot"),
        (protocol, [21000, 21001], None, "sampling", "white"),
    ]
    for args in bad:
        with pytest.raises(ValueError):
            check_reserved_request(*args)


class SequenceEnvironment(Environment):
    """Finishes with the given reward after `clicks` steps."""

    def __init__(self, clicks=2, reward=1.0):
        super().__init__(reward, True)
        self.clicks = clicks

    def step(self, action):
        self.actions.append(action)
        done = len(self.actions) >= self.clicks
        return VisionStep(self.observation, self.reward if done else 0.0, done, False)


class SequencePolicy(Policy):
    def __init__(self, texts):
        super().__init__()
        self.texts = list(texts)

    def generate(self, observation, *, seed):
        self.calls.append((observation, seed))
        return replace(self.output, text=self.texts[len(self.calls) - 1])


def click(x, y):
    return f"<tool_call><function=click><parameter=x>{x}</parameter><parameter=y>{y}</parameter></function></tool_call>"


def test_sequence_sends_history_and_succeeds_on_the_terminal_reward(tmp_path):
    from browser_control.vision_evaluation import collect_vision_sequence

    env, policy = SequenceEnvironment(2), SequencePolicy([click(100, 200), click(700, 300)])
    row = collect_vision_sequence(env, policy, 30, "screenshot", tmp_path, max_steps=4)
    assert row["success"] and row["environment_steps"] == 2 and row["stop_reason"] == "done"
    first, second = (call[0] for call in policy.calls)
    assert first.goal == env.observation.goal
    assert second.goal.endswith("Earlier clicks (x, y on 0–1000): (100, 200)")
    assert [call[1] for call in policy.calls] == [480, 481]
    assert (tmp_path / "seed-30.png").exists() and (tmp_path / "seed-30-g480-turn-1.png").exists()
    assert summarize_vision_episodes([row])["successes"] == 1


def test_sequence_stops_unrepaired_on_an_invalid_call_and_at_the_step_limit(tmp_path):
    from browser_control.vision_evaluation import collect_vision_sequence

    env = SequenceEnvironment(3)
    row = collect_vision_sequence(env, SequencePolicy([click(1, 2), "click(5)"]), 30, "screenshot", tmp_path, max_steps=4)
    assert not row["success"] and row["stop_reason"] == "invalid-call" and len(env.actions) == 1
    assert not row["valid_action"] and "error" in row["turns"][-1]
    env = SequenceEnvironment(3)
    row = collect_vision_sequence(env, SequencePolicy([click(1, 2)] * 2), 30, "screenshot", tmp_path / "b", max_steps=2)
    assert row["stop_reason"] == "step-limit" and not row["success"] and len(env.actions) == 2
    assert episode_outcome(row) == "wrong-click"


def test_samples_of_one_task_keep_separate_later_screenshots(tmp_path):
    from browser_control.vision_evaluation import collect_vision_sequence

    class Changing(SequenceEnvironment):
        def step(self, action):
            result = super().step(action)
            self.observation = observation("blue" if len(self.actions) % 2 else "green")
            return VisionStep(self.observation, result.reward, result.terminated, False)

    for member in range(2):
        env = Changing(2)
        texts = [click(100 + member, 200), click(700, 300)]
        collect_vision_sequence(env, SequencePolicy(texts), 30, "screenshot", tmp_path,
                                max_steps=4, generation_seed=1000 + member * 16)
    assert {p.name for p in tmp_path.glob("*turn-1.png")} == {"seed-30-g1000-turn-1.png", "seed-30-g1016-turn-1.png"}


def test_second_reserved_range_needs_its_own_locked_protocol():
    from browser_control.vision_evaluation import check_reserved_request

    protocol = {**reserved_protocol(), "seeds": [22000, 22001]}
    check_reserved_request(protocol, [22000, 22001], None, "sampling", "screenshot")
    with pytest.raises(ValueError):
        check_reserved_request({**protocol, "seeds": [22000, 23000]}, [22000, 23000], None, "sampling", "screenshot")
