import pytest

from browser_control.actions import InvalidActionError, parse_click_test_action


@pytest.mark.parametrize(
    ("output", "rendered"),
    [
        ("noop()", "noop()"),
        ("  click('13')  ", "click('13')"),
        ('click("a_42-x")', "click('a_42-x')"),
    ],
)
def test_valid_click_test_actions_are_canonicalized(
    output: str, rendered: str
) -> None:
    assert parse_click_test_action(output).render() == rendered


@pytest.mark.parametrize(
    "output",
    [
        "",
        "click('13')\nnoop()",
        "I would click('13')",
        "```click('13')```",
        "fill('13', 'text')",
        "send_keys('Enter')",
        "scroll('down')",
        "noop(500)",
        "click()",
        "click('13', '14')",
        "click(bid='13')",
        "click(13)",
        "click(variable)",
        "browser.click('13')",
        "__import__('os')",
        "click(compile('code', '', 'exec'))",
        "click('bad id')",
        "click('')",
    ],
)
def test_invalid_or_adversarial_outputs_are_rejected(output: str) -> None:
    with pytest.raises(InvalidActionError):
        parse_click_test_action(output)

