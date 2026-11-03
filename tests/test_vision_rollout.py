import json
from types import SimpleNamespace

import pytest

pytest.importorskip("PIL.Image")

from browser_control import vision_rollout  # noqa: E402
from browser_control.vision_rollout import (  # noqa: E402
    VisionGroupCollector,
    fit_generation_seed,
)


class Model:
    def __init__(self):
        self.training = True
        self.modes = []

    def eval(self):
        self.training = False
        self.modes.append("eval")

    def train(self):
        self.training = True
        self.modes.append("train")


def png():
    import io

    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(data, format="PNG")
    return data.getvalue()


@pytest.fixture
def fake_episode(monkeypatch, tmp_path):
    calls = []

    def collect(environment, policy, seed, condition, directory, *, generation_seed, record_logprobs):
        calls.append((seed, generation_seed, condition, record_logprobs, policy.model.training))
        (directory / f"seed-{seed}.png").write_bytes(png())
        member = len(calls) - 1
        return {
            "goal": "Click on the red box.",
            "input_image_filename": f"seed-{seed}.png",
            "success": member == 0,
            "generation": {
                "prompt_token_ids": (1, 2),
                "completion_token_ids": (3, 4, 5),
                "sampling_logprobs": (-0.1, -0.2, -0.3),
            },
        }

    monkeypatch.setattr(vision_rollout, "collect_vision_episode", collect)
    return calls


def prompts(seed, size=4):
    return [{"prompt": json.dumps({"task_seed": seed})} for _ in range(size)]


def test_group_keeps_raw_ids_and_native_rewards(fake_episode, tmp_path):
    model = Model()
    collector = VisionGroupCollector(None, SimpleNamespace(model=model), tmp_path)
    rows = collector(prompts(112), None)
    assert [row["native_success"] for row in rows] == [True, False, False, False]
    assert rows[0]["completion_ids"] == [3, 4, 5]
    assert rows[0]["env_mask"] == [1, 1, 1]
    assert rows[0]["logprobs"] == [-0.1, -0.2, -0.3]
    assert rows[0]["images"][0].mode == "RGB"
    assert collector.groups == [
        {"task_seed": 112, "native_rewards": [1.0, 0.0, 0.0, 0.0], "reward_variation": True}
    ]


def test_samples_use_distinct_seeds_in_eval_mode_and_training_resumes(fake_episode, tmp_path):
    model = Model()
    VisionGroupCollector(None, SimpleNamespace(model=model), tmp_path)(prompts(112), None)
    assert [row[1] for row in fake_episode] == [fit_generation_seed(112, m) for m in range(4)]
    assert all(row[2] == "screenshot" and row[3] and row[4] is False for row in fake_episode)
    assert model.training is True


def test_mixed_tasks_are_refused(fake_episode, tmp_path):
    with pytest.raises(ValueError, match="one task"):
        VisionGroupCollector(None, SimpleNamespace(model=Model()), tmp_path)(
            prompts(112, 2) + prompts(113, 2), None
        )
