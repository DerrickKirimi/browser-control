from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from browser_control.evaluation import PolicyOutput, evaluate_click_test
from browser_control.testing import FakeClickTestEnvironment


@dataclass
class SeededPolicy:
    outputs: dict[int, str]

    def generate(
        self, messages: list[dict[str, str]], *, seed: int
    ) -> PolicyOutput:
        assert messages[-1]["content"].endswith("Action:")
        return PolicyOutput(
            text=self.outputs[seed],
            generation_seconds=seed / 10,
            prompt_tokens=20,
            completion_tokens=3,
        )


def _factory(created: list[FakeClickTestEnvironment]):
    def create() -> FakeClickTestEnvironment:
        environment = FakeClickTestEnvironment()
        created.append(environment)
        return environment

    return create


def test_evaluation_reports_success_failure_and_invalid_output() -> None:
    created: list[FakeClickTestEnvironment] = []
    policy = SeededPolicy(
        outputs={
            1: "click('13')",
            2: "noop()",
            3: "I would click('13')",
        }
    )

    report = evaluate_click_test(_factory(created), policy, seeds=[1, 2, 3])

    assert report.success_rate == pytest.approx(1 / 3)
    assert report.invalid_action_rate == pytest.approx(1 / 3)
    assert report.mean_environment_steps == pytest.approx(2 / 3)
    assert report.mean_generation_seconds == pytest.approx(0.2)
    assert [episode.success for episode in report.episodes] == [True, False, False]
    assert report.episodes[2].parsed_action is None
    assert report.episodes[2].environment_steps == 0
    assert report.episodes[2].error
    assert created[0]._active is False


@pytest.mark.parametrize("seeds", [[], (), iter(())])
def test_evaluation_rejects_empty_seed_set(seeds: Iterable[int]) -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_click_test(
            lambda: FakeClickTestEnvironment(), SeededPolicy(outputs={}), seeds
        )


def test_report_is_json_compatible_shape() -> None:
    report = evaluate_click_test(
        lambda: FakeClickTestEnvironment(),
        SeededPolicy(outputs={23: "click('13')"}),
        [23],
    )

    payload = report.to_dict()
    assert payload["success_rate"] == 1.0
    assert payload["episodes"][0]["seed"] == 23
