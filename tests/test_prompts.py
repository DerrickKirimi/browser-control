from browser_control.environment import BrowserObservation
from browser_control.prompts import build_click_test_messages


def test_prompt_matches_click_test_action_contract() -> None:
    messages = build_click_test_messages(
        BrowserObservation(
            goal="Click the button.",
            accessibility_tree="[13] button 'Click Me!'",
        )
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "click('<bid>')" in messages[0]["content"]
    assert "noop()" in messages[0]["content"]
    assert "send_keys" not in messages[0]["content"]
    assert "scroll" not in messages[0]["content"]
    assert "Goal: Click the button." in messages[1]["content"]
    assert "[13] button 'Click Me!'" in messages[1]["content"]


def test_prompt_includes_previous_action_error() -> None:
    messages = build_click_test_messages(
        BrowserObservation(
            goal="Click the button.",
            accessibility_tree="[13] button 'Click Me!'",
            last_action_error="BrowserGym reported an action error",
        )
    )

    assert "Previous action error: BrowserGym reported an action error" in (
        messages[1]["content"]
    )
