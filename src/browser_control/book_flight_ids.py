"""Consistent structural element-ID relabeling without changing task text."""

from __future__ import annotations

import ast
import copy
import random
import re
from collections import defaultdict

TREE_ID = re.compile(r"(?m)^([ \t]*\[)([\w-]+)(\](?:[ \t]|$))")
SELECTOR_ID = re.compile(r"(\[bid\s*=\s*['\"])([\w-]+)(['\"]\])")
XML_ID = re.compile(r"(<parameter=bid>\s*)([\w-]+)(\s*</parameter>)")


def observation_parts(content: str) -> tuple[str, str, str]:
    return tuple(content.partition("Accessibility tree:\n"))


def structural_ids(probe: dict) -> set[str]:
    found = set()
    for message in probe["messages"]:
        content = message.get("content") or ""
        header, marker, tree = observation_parts(content)
        if marker:
            found.update(m[2] for m in TREE_ID.finditer(tree))
            error = header.partition("Last action error: ")[2]
            found.update(m[2] for m in SELECTOR_ID.finditer(error))
        if message["role"] == "assistant":
            found.update(m[2] for m in XML_ID.finditer(content))
        for call in message.get("tool_calls", []):
            bid = call["function"]["arguments"].get("bid")
            if bid is not None:
                found.add(bid)
    found.update(
        call["parameters"]["bid"]
        for call in probe["accepted"]
        if "bid" in call["parameters"]
    )
    return found


def id_permutation(ids: set[str], *, seed: int) -> dict[str, str]:
    """Derange within character-width groups; singleton groups stay fixed.

    Reuse exactly the observed symbol vocabulary, preserving ID character widths
    to reduce irrelevant out-of-distribution changes. Every reference to one ID
    uses the same mapping across the entire probe conversation.
    """
    groups = defaultdict(list)
    for bid in sorted(ids):
        groups[len(bid)].append(bid)
    rng = random.Random(seed)
    mapping = {}
    for group in groups.values():
        shuffled = list(group)
        if len(group) > 1:
            while any(a == b for a, b in zip(group, shuffled, strict=True)):
                rng.shuffle(shuffled)
        mapping.update(zip(group, shuffled, strict=True))
    return mapping


def relabel_probe(probe: dict, mapping: dict[str, str]) -> dict:
    if structural_ids(probe) - mapping.keys():
        raise ValueError("Mapping must cover every structural ID")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("ID mapping must be one-to-one")
    result = copy.deepcopy(probe)

    def replace(match):
        return match[1] + mapping[match[2]] + match[3]

    for message in result["messages"]:
        if isinstance(message.get("content"), str):
            content = message["content"]
            header, marker, tree = observation_parts(content)
            if marker:
                lead, error_marker, error = header.partition("Last action error: ")
                header = lead + error_marker + SELECTOR_ID.sub(replace, error)
                content = header + marker + TREE_ID.sub(replace, tree)
            if message["role"] == "assistant":
                content = XML_ID.sub(replace, content)
            message["content"] = content
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if "bid" in arguments:
                arguments["bid"] = mapping[arguments["bid"]]
    for call in result["accepted"]:
        if "bid" in call["parameters"]:
            call["parameters"]["bid"] = mapping[call["parameters"]["bid"]]
    if "solver_action" in result:
        expression = ast.parse(result["solver_action"], mode="eval")
        call = expression.body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError("Expected a simple canonical solver action")
        if call.func.id not in {"click", "fill", "press", "noop"}:
            raise ValueError("Unexpected solver action")
        if call.args:
            bid = ast.literal_eval(call.args[0])
            call.args[0] = ast.Constant(mapping[bid])
        result["solver_action"] = ast.unparse(expression)
    return result


def compare_id_scores(original: dict, permuted: dict) -> dict:
    """Predeclared >=4/20 drop gate; a small drop is not proof of grounding."""
    rows = {}
    for skill in ("select_origin", "select_destination", "open_calendar", "search"):
        before, after = original["skills"][skill], permuted["skills"][skill]
        if before["count"] != 20 or after["count"] != 20:
            raise ValueError("ID-sensitivity gate requires 20 matched probes per skill")
        rows[skill] = {
            "original_correct": before["correct"],
            "permuted_correct": after["correct"],
            "drop": before["correct"] - after["correct"],
        }
    stop = any(row["drop"] >= 4 for row in rows.values())
    return {
        "route": "id-sensitivity-stop" if stop else "no-large-drop-on-these-probes",
        "stop_grpo": stop,
        "skills": rows,
        "scope": "ID renumbering sensitivity; does not by itself establish a memorization mechanism or general grounding",
    }
