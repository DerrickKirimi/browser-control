"""Strict policy-output parsing for BrowserGym actions."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_BID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InvalidActionError(ValueError):
    """Raised when model output is not one permitted action."""


@dataclass(frozen=True, slots=True)
class ParsedAction:
    """A validated action with a stable BrowserGym representation."""

    name: str
    arguments: tuple[str, ...] = ()

    def render(self) -> str:
        arguments = ", ".join(repr(argument) for argument in self.arguments)
        return f"{self.name}({arguments})"


def parse_click_test_action(output: str) -> ParsedAction:
    """Parse exactly one `noop()` or `click('<bid>')` expression."""

    candidate = output.strip()
    if not candidate:
        raise InvalidActionError("model output is empty")
    if "\n" in candidate or "\r" in candidate:
        raise InvalidActionError("model output must contain exactly one line")

    try:
        expression = ast.parse(candidate, mode="eval")
    except SyntaxError as error:
        raise InvalidActionError("model output is not a valid action expression") from error

    call = expression.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise InvalidActionError("model output must be a direct action call")
    if call.keywords:
        raise InvalidActionError("keyword arguments are not allowed")

    if call.func.id == "noop":
        if call.args:
            raise InvalidActionError("noop takes no arguments")
        return ParsedAction(name="noop")

    if call.func.id != "click":
        raise InvalidActionError(f"action {call.func.id!r} is not allowed for click-test")
    if len(call.args) != 1:
        raise InvalidActionError("click requires exactly one browser element ID")

    argument = call.args[0]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        raise InvalidActionError("the click element ID must be a string literal")
    if not _BID_PATTERN.fullmatch(argument.value):
        raise InvalidActionError("the click element ID has an invalid format")
    return ParsedAction(name="click", arguments=(argument.value,))
