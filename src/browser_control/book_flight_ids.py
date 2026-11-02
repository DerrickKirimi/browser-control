"""Consistent structural element-ID relabeling without changing task text."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

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


def compare_id_scores(original: dict, permuted: dict, *, skills=None) -> dict:
    """Predeclared >=4/20 drop gate; a small drop is not proof of grounding."""
    from browser_control.book_flight_diagnostic import DECISION_SKILLS

    if skills is None:
        skills = ("select_origin", "select_destination", "open_calendar", "search")
    if (
        not skills
        or len(set(skills)) != len(skills)
        or set(skills) - set(DECISION_SKILLS)
    ):
        raise ValueError("ID gate requires distinct known critical skills")
    rows = {}
    for skill in skills:
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


def augment_training_examples(
    examples: list[dict], *, seed: int
) -> tuple[list[dict], dict]:
    """Relabel each independent SFT conversation and current target consistently.

    Raw solver records stay unchanged. Each example uses its task/turn to draw
    reproducibly, so shuffling does not change its augmentation. Only training
    seeds are accepted; this is not a transformation of evaluation evidence.
    """
    from browser_control.book_flight_sft import TRAINING_SEEDS

    augmented, hashes = [], []
    for example in examples:
        if example["task_seed"] not in TRAINING_SEEDS:
            raise ValueError("ID-augmented SFT accepts only training seeds")
        draw_seed = seed + example["task_seed"] * 16 + example["turn"]
        surrogate = {
            "messages": [*example["messages"], example["assistant"]],
            "accepted": [],
        }
        mapping = id_permutation(structural_ids(surrogate), seed=draw_seed)
        altered = relabel_probe(surrogate, mapping)
        target = example["assistant"]["tool_calls"][0]["function"]["arguments"].get(
            "bid"
        )
        if target is not None and mapping[target] == target:
            raise ValueError("Training target ID was not randomized")
        digest = hashlib.sha256(
            json.dumps(mapping, sort_keys=True).encode()
        ).hexdigest()
        augmented.append(
            {
                **copy.deepcopy(example),
                "messages": altered["messages"][:-1],
                "assistant": altered["messages"][-1],
                "id_mapping_sha256": digest,
                "id_permutation_seed": draw_seed,
            }
        )
        hashes.append((example["task_seed"], example["turn"], digest))
    policy = {
        "type": "structural ID permutation within character widths",
        "seed": seed,
        "draw_seed": "seed + task_seed * 16 + turn",
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "mappings_sha256": hashlib.sha256(
            json.dumps(sorted(hashes)).encode()
        ).hexdigest(),
    }
    return augmented, policy


def validate_probe_pair(original: dict, altered: dict, *, original_sha256: str) -> None:
    """Require the declared structural-only, width-preserving paired control."""
    if altered["source_sha256"] != original_sha256:
        raise ValueError("Permuted probes do not match the exact original source")
    rows = original["probes"]
    if len(rows) != len(altered["probes"]) or len(rows) != len(altered["mappings"]):
        raise ValueError("Paired probe lengths differ")
    for index, (before, after) in enumerate(zip(rows, altered["probes"], strict=True)):
        mapping_row = altered["mappings"][index]
        mapping = mapping_row["mapping"]
        ids = structural_ids(before)
        if mapping_row["probe_index"] != index:
            raise ValueError("Permutation mapping order differs from probe order")
        if set(mapping) != ids or set(mapping.values()) != ids:
            raise ValueError(
                "Permutation must reuse the exact structural ID vocabulary"
            )
        if any(len(key) != len(value) for key, value in mapping.items()):
            raise ValueError("Permutation must preserve ID character widths")
        if mapping != id_permutation(ids, seed=altered["permutation_seed"] + index):
            raise ValueError("Permutation does not match its declared draw seed")
        if any(
            mapping[call["parameters"]["bid"]] == call["parameters"]["bid"]
            for call in before["accepted"]
            if "bid" in call["parameters"]
        ):
            raise ValueError("An accepted target ID was not permuted")
        if relabel_probe(before, mapping) != after:
            raise ValueError("Permuted probe changes data outside structural IDs")
