import pytest

from browser_control.environment import BrowserEnvironment
from browser_control.testing import FakeClickTestEnvironment


def test_fake_environment_satisfies_runtime_protocol() -> None:
    environment = FakeClickTestEnvironment()

    assert isinstance(environment, BrowserEnvironment)


def test_expected_click_terminates_with_success() -> None:
    environment = FakeClickTestEnvironment(button_id="42")

    initial = environment.reset(seed=23)
    result = environment.step("click('42')")

    assert not initial.done
    assert result.done
    assert result.reward == 1.0
    assert result.info["success"] is True
    assert environment.actions == ["click('42')"]


def test_wrong_action_terminates_without_reward() -> None:
    environment = FakeClickTestEnvironment()
    environment.reset()

    result = environment.step("noop()")

    assert result.done
    assert result.reward == 0.0
    assert result.observation.last_action_error == "invalid action: noop()"


def test_step_requires_an_active_episode() -> None:
    environment = FakeClickTestEnvironment()

    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        environment.step("noop()")


def test_step_after_terminal_result_is_rejected() -> None:
    environment = FakeClickTestEnvironment()
    environment.reset()
    environment.step(environment.expected_action)

    with pytest.raises(RuntimeError, match=r"after the episode is done"):
        environment.step(environment.expected_action)

