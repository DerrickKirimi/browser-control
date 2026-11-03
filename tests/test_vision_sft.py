import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from collect_lfm3_vision_experts import BOX_CENTRES, label_task  # noqa: E402
from train_lfm3_vision_sft import canonical_call, schedule  # noqa: E402

from browser_control.vision_browser import parse_coordinate_click  # noqa: E402


def test_canonical_call_parses_to_its_target():
    parsed = parse_coordinate_click(canonical_call(147, 781))
    assert (parsed.x, parsed.y) == (147, 781)


def test_schedule_uses_each_task_once_per_pass():
    examples = [{"task_seed": seed} for seed in range(5)]
    order = schedule(examples, 4, 23)
    assert len(order) == 20
    for start in range(0, 20, 5):
        assert sorted(e["task_seed"] for e in order[start : start + 5]) == list(range(5))
    assert order == schedule(examples, 4, 23)


class Step:
    def __init__(self, reward):
        self.reward = reward
        self.done = True
        self.observation = "obs"


class Env:
    def __init__(self, winners):
        self.winners = winners
        self.resets = 0

    def reset(self, seed):
        self.resets += 1
        return Step(0)

    def step(self, click):
        return Step(1.0 if (click.x, click.y) in self.winners else 0.0)


def test_label_task_keeps_only_rewarded_centres_and_resets_each_time():
    env = Env({BOX_CENTRES[2]})
    observation, winners = label_task(env, 120)
    assert observation == "obs"
    assert winners == [{"x": BOX_CENTRES[2][0], "y": BOX_CENTRES[2][1]}]
    assert env.resets == 5
