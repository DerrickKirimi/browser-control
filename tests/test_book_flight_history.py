"""Alternative token boundaries must preserve actual IDs and exact context text."""

import runpy
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from browser_control.environment import BrowserObservation

module = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts/check_book_flight_token_rollouts.py"
    )
)
audit_history = module["audit_history"]
history_text_matches = module["history_text_matches"]
diagnostic_history_matches = module["diagnostic_history_matches"]
FILL = "<tool_call><function=fill><parameter=bid>19</parameter><parameter=value>GNU</parameter></function></tool_call>"
CLICK = "<tool_call><function=click><parameter=bid>1</parameter></function></tool_call>"


class Tokenizer:
    eos_token_id = 99

    def decode(self, ids, **kwargs):
        prefix, suffix = FILL.split("GNU")
        pieces = {
            10: "prompt",
            11: ":",
            42: FILL,
            43: prefix + "G",
            44: "NU" + suffix,
            45: CLICK,
            46: "BAD" + suffix,
            99: "<eos>",
            7: "\n",
            55: "observation",
            56: ":",
            57: "changed",
            77: "<assistant>",
        }
        return "".join(pieces[token] for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        result = [10, 11]
        assistants = 0
        for message in messages:
            if message["role"] == "assistant":
                result.extend([42 if assistants == 0 else 45, 99, 7])
                assistants += 1
            elif message["role"] == "tool":
                result.extend([57 if "changed" in message["content"] else 55, 56, 77])
        return {"input_ids": result}


def episode():
    observation = asdict(BrowserObservation("goal", "[1] button"))
    return {
        "prompt_ids": [10, 11],
        "completion_ids": [43, 44, 99, 7, 55, 56, 77, 45, 99],
        "steps": [
            {
                "turn": 0,
                "observation": observation,
                "text": FILL,
                "token_ids": [43, 44, 99],
                "prompt_tokens": 2,
                "valid_action": True,
            },
            {
                "turn": 1,
                "observation": deepcopy(observation),
                "text": CLICK,
                "token_ids": [45, 99],
                "prompt_tokens": 9,
                "valid_action": True,
            },
        ],
    }


def test_identical_text_with_different_boundaries_keeps_sampled_ids():
    item = episode()
    original = deepcopy(item)
    rows = audit_history(item, Tokenizer())
    assert not rows[0]["assistant_ids_match"]
    assert not rows[1]["context_ids_match"]
    assert history_text_matches(rows)
    assert item == original


def test_changed_sampled_text_is_rejected():
    item = episode()
    item["completion_ids"][1] = 46
    item["steps"][0]["token_ids"][1] = 46
    assert not history_text_matches(audit_history(item, Tokenizer()))


def test_changed_external_feedback_is_rejected():
    item = episode()
    item["completion_ids"][4] = 57
    assert not history_text_matches(audit_history(item, Tokenizer()))


def test_changed_observed_state_is_rejected():
    item = episode()
    item["steps"][1]["observation"]["accessibility_tree"] = "changed"
    assert not history_text_matches(audit_history(item, Tokenizer()))


def test_empty_history_does_not_count_as_verified():
    assert not history_text_matches([])


def test_learning_diagnostic_accepts_alternative_boundaries_but_contract_does_not():
    item = episode()
    original = deepcopy(item)
    item["history_audit"] = audit_history(item, Tokenizer())
    diagnostic = {"episodes": [item], "canonical_history_ids_match": False}
    assert diagnostic_history_matches(diagnostic, require_canonical=False)
    assert not diagnostic_history_matches(diagnostic, require_canonical=True)
    assert {key: value for key, value in item.items() if key != "history_audit"} == original


def test_diagnostic_checks_every_episode_instead_of_trusting_summary():
    good = episode()
    good["history_audit"] = audit_history(good, Tokenizer())
    bad = episode()
    bad["completion_ids"][4] = 57
    bad["history_audit"] = audit_history(bad, Tokenizer())
    diagnostic = {"episodes": [good, bad], "canonical_history_ids_match": True}
    assert not diagnostic_history_matches(diagnostic, require_canonical=False)
    for episodes in ([], [{}], [good, {}]):
        assert not diagnostic_history_matches(
            {"episodes": episodes}, require_canonical=False
        )


def test_canonical_diagnostic_passes_both_modes():
    item = episode()
    item["completion_ids"][:2] = [42]
    item["steps"][0]["token_ids"] = [42, 99]
    item["steps"][1]["prompt_tokens"] = 8
    item["history_audit"] = audit_history(item, Tokenizer())
    for require_canonical in (False, True):
        assert diagnostic_history_matches(
            {"episodes": [item]}, require_canonical=require_canonical
        )


def test_legacy_canonical_ids_prove_text_identity_without_new_text_fields():
    rows = [{"context_ids_match": True, "assistant_ids_match": True}]
    diagnostic = {"episodes": [{"history_audit": rows}]}
    assert diagnostic_history_matches(diagnostic, require_canonical=True)
    rows[0]["assistant_ids_match"] = False
    assert not diagnostic_history_matches(diagnostic, require_canonical=False)
    rows[0]["assistant_text_match"] = True
    assert diagnostic_history_matches(diagnostic, require_canonical=False)
    rows[0]["context_text_match"] = False
    assert not diagnostic_history_matches(diagnostic, require_canonical=False)
