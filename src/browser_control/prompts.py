"""Policy prompts derived from the validated action contract."""

from __future__ import annotations

from browser_control.environment import BrowserObservation

CLICK_TEST_SYSTEM_PROMPT = """You control a browser to complete a single task.

The page is represented as an accessibility tree. Interactive elements have a
browser element ID in brackets, for example: [13] button 'Click Me!'

Return exactly one action and no other text. The only valid actions are:
- click('<bid>') — click the element with that browser element ID
- noop() — wait without interacting

Example: click('13')"""


def build_click_test_messages(
    observation: BrowserObservation,
) -> list[dict[str, str]]:
    """Build one conversational policy input without lossy tree truncation."""

    error_section = ""
    if observation.last_action_error:
        error_section = f"\nPrevious action error: {observation.last_action_error}\n"
    user_prompt = (
        f"Goal: {observation.goal}\n"
        f"{error_section}"
        "Accessibility tree:\n"
        f"{observation.accessibility_tree}\n\n"
        "Action:"
    )
    return [
        {"role": "system", "content": CLICK_TEST_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
