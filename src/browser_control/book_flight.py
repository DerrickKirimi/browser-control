"""Bounded native actions and conversational context for multi-step book-flight."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from browser_control.actions import InvalidActionError, parse_click_test_action
from browser_control.environment import BrowserObservation
from browser_control.lfm3 import TOOLS

KEYS = {
    "ArrowDown",
    "ArrowUp",
    "ArrowLeft",
    "ArrowRight",
    "Enter",
    "Tab",
    "Escape",
    "Backspace",
}


def _tool(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


BOOK_FLIGHT_TOOLS = [
    *TOOLS,
    _tool(
        "fill",
        "Fill an editable field and trigger its autocomplete menu. "
        "Use click on calendar controls for read-only date fields.",
        {
            "bid": {"type": "string", "description": "Browser element ID"},
            "value": {
                "type": "string",
                "description": "Text to enter, at most 256 characters",
            },
        },
    ),
    _tool(
        "press",
        "Press a navigation key in a focused element.",
        {
            "bid": {"type": "string", "description": "Browser element ID"},
            "key": {"type": "string", "enum": sorted(KEYS)},
        },
    ),
]
SYSTEM_PROMPT = (
    "Complete the browser task using exactly one provided tool per turn. "
    "Return only the tool call, with no reasoning or prose. "
    "Use browser element IDs from the current accessibility tree. "
    "Observations after each action show the updated page. "
    "Use autocomplete suggestions to select full airport names. "
    "Read-only date fields require calendar navigation. "
    "Compare the flight results according to the requested criterion before booking."
)


@dataclass(frozen=True)
class BookFlightAction:
    name: str
    parameters: dict[str, str]
    reasoning: str | None = None

    def render(self) -> str:
        if self.name in {"click", "noop"}:
            expression = (
                "noop()"
                if self.name == "noop"
                else f"click({self.parameters['bid']!r})"
            )
            return parse_click_test_action(expression).render()
        bid = self.parameters["bid"]
        parse_click_test_action(f"click({bid!r})")
        if self.name == "fill":
            # Third argument triggers the real autocomplete UI, without DOM mutation.
            return f"fill({bid!r}, {self.parameters['value']!r}, True)"
        return f"press({bid!r}, {self.parameters['key']!r})"

    def message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": self.name, "arguments": self.parameters},
                }
            ],
        }
        if self.reasoning is not None:
            message["reasoning_content"] = self.reasoning
        return message


def parse_book_flight_action(output: str) -> BookFlightAction:
    match = re.fullmatch(
        r"\s*<tool_call>\s*<function=(click|noop|fill|press)>\s*(.*?)\s*</function>\s*</tool_call>\s*",
        output,
        flags=re.DOTALL,
    )
    if match is None:
        raise InvalidActionError("expected exactly one permitted XML tool call")
    name, body = match.groups()
    parameters = {}
    remaining = body.strip()
    while remaining:
        parameter = re.match(
            r"<parameter=(\w+)>\s*([^<>]*?)\s*</parameter>", remaining, flags=re.DOTALL
        )
        if parameter is None or parameter[1] in parameters:
            raise InvalidActionError("malformed or duplicate tool parameter")
        parameters[parameter[1]] = parameter[2].strip()
        remaining = remaining[parameter.end() :].strip()
    required = {
        "noop": set(),
        "click": {"bid"},
        "fill": {"bid", "value"},
        "press": {"bid", "key"},
    }[name]
    if set(parameters) != required:
        raise InvalidActionError(f"{name} requires exactly {sorted(required)}")
    if name == "fill" and (
        len(parameters["value"]) > 256 or any(ord(c) < 32 for c in parameters["value"])
    ):
        raise InvalidActionError("fill value must be at most 256 printable characters")
    if name == "press" and parameters["key"] not in KEYS:
        raise InvalidActionError("unsupported navigation key")
    action = BookFlightAction(name, parameters)
    action.render()  # Apply the shared element-ID validator before execution.
    return action


def parse_reasoned_action(output: str) -> BookFlightAction:
    """Accept reasoning closed by one </think>, then exactly one permitted call."""
    if output.count("</think>") != 1:
        raise InvalidActionError("expected reasoning closed by exactly one </think>")
    reasoning, call = output.split("</think>")
    return replace(parse_book_flight_action(call), reasoning=reasoning)


def observation_text(observation: BrowserObservation) -> str:
    error = observation.last_action_error or "none"
    return f"Goal: {observation.goal}\nLast action error: {error}\nAccessibility tree:\n{observation.accessibility_tree}"


def initial_messages(
    observation: BrowserObservation, system_prompt: str = SYSTEM_PROMPT
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation_text(observation)},
    ]
