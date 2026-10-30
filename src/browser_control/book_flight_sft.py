"""Training-only expert examples and current-action-only causal labels."""

from __future__ import annotations

from typing import Any

from browser_control.book_flight import BOOK_FLIGHT_TOOLS

TRAINING_SEEDS = range(100, 10000)


def expert_examples(record: dict) -> list[dict]:
    """Reject whole error/retry histories, including timeouts with visible effects."""
    seed = record["task_seed"]
    if seed not in TRAINING_SEEDS:
        raise ValueError("Expert SFT accepts only training seeds 100..9999")
    if not record["success"] or record["reward"] <= 0:
        return []
    if not record["steps"] or any(step["error"] for step in record["steps"]):
        return []
    examples = []
    for index, step in enumerate(record["steps"]):
        if step["observation"].get("last_action_error"):
            return []
        assistant = step["assistant"]
        if assistant["role"] != "assistant" or len(assistant["tool_calls"]) != 1:
            raise ValueError("Expected exactly one canonical expert tool call")
        examples.append(
            {
                "task_seed": seed,
                "turn": index,
                "skill": step["skill"],
                "messages": step["messages"],
                "assistant": assistant,
            }
        )
    return examples


def encode_example(tokenizer: Any, example: dict, *, max_tokens: int = 16384) -> dict:
    """Mask the entire prompt; supervise only the next canonical call and EOS.

    Earlier assistant turns and all tool observations remain context with -100
    labels. Prefix equality is checked rather than inferred from string lengths.
    No packing, padding or truncation changes the conversation boundary.
    """
    kwargs = dict(
        tools=BOOK_FLIGHT_TOOLS, enable_thinking=False, tokenize=True, return_dict=True
    )
    prompt = tokenizer.apply_chat_template(
        example["messages"], **kwargs, add_generation_prompt=True
    )["input_ids"]
    full = tokenizer.apply_chat_template(
        [*example["messages"], example["assistant"]],
        **kwargs,
        add_generation_prompt=False,
    )["input_ids"]
    if full[: len(prompt)] != prompt:
        raise ValueError("Chat template prefix mismatch")
    try:
        end = full.index(tokenizer.eos_token_id, len(prompt)) + 1
    except ValueError as error:
        raise ValueError("Expected a current call ending in EOS") from error
    # LFM3's template adds a newline after EOS. Generation stops at EOS, so
    # remove only this template whitespace rather than teaching a post-EOS token.
    if tokenizer.decode(full[end:], skip_special_tokens=False).strip():
        raise ValueError("Unexpected non-whitespace tokens after current-call EOS")
    full = full[:end]
    if end <= len(prompt) + 1:
        raise ValueError("Expected a nonempty current call before EOS")
    if len(full) > max_tokens:
        raise ValueError(f"Expert example exceeds {max_tokens} tokens; no truncation")
    return {
        "input_ids": full,
        "labels": [-100] * len(prompt) + full[len(prompt) :],
        "prompt_tokens": len(prompt),
        "target_tokens": len(full) - len(prompt),
    }
