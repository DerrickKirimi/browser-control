from __future__ import annotations

import pytest

from browser_control.rewards import build_click_test_rows, make_click_test_reward
from browser_control.testing import FakeClickTestEnvironment


def test_training_rows_freeze_seeded_prompts_and_close_environments() -> None:
    created: list[FakeClickTestEnvironment] = []

    def factory() -> FakeClickTestEnvironment:
        environment = FakeClickTestEnvironment()
        created.append(environment)
        return environment

    rows = build_click_test_rows(factory, (23, 24))

    assert [row["task_seed"] for row in rows] == [23, 24]
    assert all(row["prompt"][-1]["content"].endswith("Action:") for row in rows)
    assert all(not environment._active for environment in created)


def test_reward_executes_only_strictly_valid_actions() -> None:
    created: list[FakeClickTestEnvironment] = []

    def factory() -> FakeClickTestEnvironment:
        environment = FakeClickTestEnvironment()
        created.append(environment)
        return environment

    reward = make_click_test_reward(factory)
    rewards = reward(
        ["click('13')", "noop()", "click(<button id='13'>)"],
        [23, 23, 23],
    )

    assert rewards == [1.0, 0.0, 0.0]
    assert len(created) == 2
    assert [environment.actions for environment in created] == [
        ["click('13')"],
        ["noop()"],
    ]
    assert all(not environment._active for environment in created)
    assert [record["valid_action"] for record in reward.records] == [True, True, False]


def test_reward_rejects_misaligned_metadata() -> None:
    reward = make_click_test_reward(lambda: FakeClickTestEnvironment())
    with pytest.raises(ValueError, match="equal lengths"):
        reward(["noop()"], [23, 24])


def test_reward_accepts_trl_conversational_completion_shape() -> None:
    reward = make_click_test_reward(lambda: FakeClickTestEnvironment())
    assert reward([[{"role": "assistant", "content": "click('13')"}]], [23]) == [
        1.0
    ]


@pytest.mark.parametrize(
    "completion",
    [[], [{"role": "user", "content": "noop()"}], [{"role": "assistant"}]],
)
def test_reward_rejects_ambiguous_completion_shapes(completion: object) -> None:
    reward = make_click_test_reward(lambda: FakeClickTestEnvironment())
    with pytest.raises(TypeError, match="plain text or one assistant"):
        reward([completion], [23])


def test_training_rows_reject_empty_seed_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_click_test_rows(lambda: FakeClickTestEnvironment(), ())


def test_format_curriculum_produces_auditable_reward_levels() -> None:
    reward = make_click_test_reward(
        lambda: FakeClickTestEnvironment(), format_curriculum=True
    )

    rewards = reward(
        ["click('13')", "noop()", 'click(<button id="13">Click Me!</button>)', "junk"],
        [23, 23, 23, 23],
    )

    assert rewards[0] == 1.0
    assert rewards[1] == 0.25
    assert 0.0 <= rewards[3] < rewards[2] < 0.25
    assert [record["completion"] for record in reward.records] == [
        "click('13')",
        "noop()",
        'click(<button id="13">Click Me!</button>)',
        "junk",
    ]
