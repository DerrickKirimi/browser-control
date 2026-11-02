"""Prevent seed leakage and loss on historical assistant/tool tokens."""

from copy import deepcopy

import pytest

from browser_control.book_flight_sft import encode_example, expert_examples


def record():
    return {
        "task_seed": 100,
        "success": True,
        "reward": 1.0,
        "steps": [
            {
                "skill": "select_origin",
                "error": "",
                "observation": {"last_action_error": ""},
                "messages": [
                    {"role": "system", "content": "instructions"},
                    {"role": "assistant", "content": "earlier call"},
                    {"role": "tool", "content": "airport suggestions"},
                ],
                "assistant": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "click", "arguments": {"bid": "34"}},
                        }
                    ],
                },
            }
        ],
    }


@pytest.mark.parametrize("seed", [0, 30, 99, 10000, 10100, 10299])
def test_only_training_seeds_are_accepted(seed):
    item = record()
    item["task_seed"] = seed
    with pytest.raises(ValueError, match="training seeds"):
        expert_examples(item)


def test_timeout_excludes_entire_episode_even_when_solver_succeeded():
    item = record()
    item["steps"].append(deepcopy(item["steps"][0]))
    item["steps"][1]["error"] = "Timeout: click may have taken effect"
    assert expert_examples(item) == []


def test_successful_clean_episode_keeps_current_canonical_call():
    (example,) = expert_examples(record())
    assert example["skill"] == "select_origin"
    assert example["assistant"]["tool_calls"][0]["function"]["name"] == "click"


class Tokenizer:
    eos_token_id = 99

    def decode(self, tokens, **kwargs):
        return "" if not tokens else "\n" if tokens == [22] else "unexpected"

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        assert not kwargs["enable_thinking"]
        assert kwargs["tools"]
        assert kwargs["return_dict"]
        return {
            "input_ids": [10, 11, 12, 13]
            if add_generation_prompt
            else [10, 11, 12, 13, 20, 21, 99]
        }


def test_only_current_call_and_eos_receive_labels():
    encoded = encode_example(Tokenizer(), expert_examples(record())[0])
    assert encoded["input_ids"] == [10, 11, 12, 13, 20, 21, 99]
    assert encoded["labels"] == [-100, -100, -100, -100, 20, 21, 99]
    assert encoded["target_tokens"] == 3


def test_long_examples_fail_instead_of_truncating():
    with pytest.raises(ValueError, match="no truncation"):
        encode_example(Tokenizer(), expert_examples(record())[0], max_tokens=6)


def test_template_newline_after_eos_is_not_supervised():
    class NewlineTokenizer(Tokenizer):
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            return {
                "input_ids": (
                    [10, 11, 12, 13]
                    if add_generation_prompt
                    else [10, 11, 12, 13, 20, 99, 22]
                )
            }

    result = encode_example(NewlineTokenizer(), expert_examples(record())[0])
    assert result["labels"] == [-100, -100, -100, -100, 20, 99]


@pytest.mark.parametrize("full", [[10, 11, 22, 13, 20, 99], [10, 11, 12, 13, 20]])
def test_prefix_changes_and_missing_eos_are_rejected(full):
    class BrokenTokenizer(Tokenizer):
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            return {"input_ids": [10, 11, 12, 13] if add_generation_prompt else full}

    with pytest.raises(ValueError):
        encode_example(BrokenTokenizer(), expert_examples(record())[0])
