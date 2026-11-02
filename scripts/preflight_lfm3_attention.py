"""Compare SDPA with eager on development prompts, then test one LoRA update."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from browser_control.book_flight import BOOK_FLIGHT_TOOLS, BookFlightAction
from browser_control.lfm3 import LFM3Policy
from browser_control.transformers_policy import DecoderConfig

MAX_LOGIT_DIFFERENCE = 0.1
LONG_PROMPT_TOKENS = 12000
LORA = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj",
        "out_proj",
        "w1",
        "w2",
        "w3",
    ],
    "task_type": "CAUSAL_LM",
}


def load(attention: str) -> LFM3Policy:
    policy = LFM3Policy.load(
        DecoderConfig("greedy", False, 2), "native-tools", attention
    )
    policy.tools = BOOK_FLIGHT_TOOLS
    policy.max_prompt_tokens = None
    return policy


def release(policy: LFM3Policy) -> None:
    torch = policy.torch
    policy.model = None
    gc.collect()
    torch.cuda.empty_cache()


def forward(policy: LFM3Policy, prompts: list[list[dict]]) -> list[dict]:
    torch = policy.torch
    rows = []
    for messages in prompts:
        torch.cuda.reset_peak_memory_stats()
        encoded = policy.encode(messages)
        started = time.monotonic()
        with torch.inference_mode():
            logits = (
                policy.model(**encoded, logits_to_keep=1).logits[:, -1, :].float().cpu()
            )
        rows.append(
            {
                "prompt_tokens": int(encoded["input_ids"].shape[-1]),
                "logits": logits,
                "finite": bool(torch.isfinite(logits).all().item()),
                "seconds": time.monotonic() - started,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    return rows


def lora_attempt(
    policy, model, ids, target_length, checkpointing, *, initial_scale=65536.0
) -> dict:
    """One real scaled backward and optimizer step; the adapter is discarded."""
    torch = policy.torch
    row = {
        "prompt_tokens": int(ids.shape[-1]),
        "checkpointing": checkpointing,
        "initial_scale": initial_scale,
        "update_passed": False,
    }
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n for n in trainable):
        raise ValueError("Only LoRA parameters may be trainable")
    before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    optimizer = torch.optim.SGD(trainable.values(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", init_scale=initial_scale)
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    gradients = []
    logits = loss = norm = None
    try:
        if checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.gradient_checkpointing_disable()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                logits_to_keep=target_length + 1,
            ).logits
            loss = torch.nn.functional.cross_entropy(
                logits[0, :-1].float(), ids[0, -target_length:]
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in trainable.values() if p.grad is not None]
        norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.float()) for g in gradients])
        )
        row.update(
            loss=loss.item(),
            gradient_norm=norm.item(),
            tensors_with_gradient=len(gradients),
            nonfinite_gradient_tensors=sum(
                not torch.isfinite(g).all().item() for g in gradients
            ),
            nonzero_gradient_tensors=sum(bool(g.any().item()) for g in gradients),
            nonfinite_gradient_names=[
                n
                for n, p in trainable.items()
                if p.grad is not None and not torch.isfinite(p.grad).all().item()
            ],
        )
        scaler.step(optimizer)
        scaler.update()
        row.update(
            status="completed",
            changed_tensors=sum(
                not torch.equal(p.detach().cpu(), before[n])
                for n, p in trainable.items()
            ),
            seconds=time.monotonic() - started,
            final_scale=scaler.get_scale(),
            parameters_finite=all(
                bool(torch.isfinite(p).all().item()) for p in trainable.values()
            ),
        )
        row["update_passed"] = bool(
            torch.isfinite(loss).item()
            and torch.isfinite(norm).item()
            and row["gradient_norm"] > 0
            and row["tensors_with_gradient"] == len(trainable)
            and not row["nonfinite_gradient_tensors"]
            and row["parameters_finite"]
            and row["changed_tensors"] > 0
        )
    except torch.OutOfMemoryError as error:
        row.update(status="out-of-memory", error=str(error)[:300])
    except (RuntimeError, ValueError, TypeError, AttributeError) as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}"[:600])
    finally:
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(before[name].to(parameter.device))
    row["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    print({k: v for k, v in row.items() if k != "nonfinite_gradient_names"}, flush=True)
    logits = loss = norm = gradients = None
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return row


def lora_stage(selected, report, longest, *, tree_source, output) -> None:
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(selected.model, LoraConfig(**LORA))
    model.train()
    report["lora"] = {
        "config": LORA,
        "trainable_tensors": sum(p.requires_grad for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "adapter_dtypes": sorted(
            {str(p.dtype) for p in model.parameters() if p.requires_grad}
        ),
    }
    action = BookFlightAction("click", {"bid": "392"}).message()
    tree = tree_source["steps"][5]["observation"]["accessibility_tree"]
    stress = list(longest)
    for repeats in range(1, 30):
        stress = [
            *longest,
            {
                "role": "user",
                "content": "Synthetic memory stress context:\n"
                + (tree + "\n") * repeats,
            },
        ]
        if selected.encode(stress)["input_ids"].shape[-1] >= LONG_PROMPT_TOKENS:
            break
    attempts = []
    for prompt in (longest, stress):
        prompt_ids = selected.encode(prompt)["input_ids"]
        full = selected.tokenizer.apply_chat_template(
            [*prompt, action],
            tools=BOOK_FLIGHT_TOOLS,
            enable_thinking=False,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"].to("cuda:0")
        target_length = full.shape[-1] - prompt_ids.shape[-1]
        prefix_matches = selected.torch.equal(
            full[:, : prompt_ids.shape[-1]], prompt_ids
        )
        for checkpointing in (False, True):
            row = lora_attempt(selected, model, full, target_length, checkpointing)
            row.update(target_tokens=target_length, prefix_matches=prefix_matches)
            attempts.append(row)
            report["lora"]["attempts"] = attempts
            output.write_text(json.dumps(report, indent=2) + "\n")
    report["lora"]["update_passed"] = any(
        row["status"] == "completed"
        and row["gradient_norm"] > 0
        and not row["nonfinite_gradient_tensors"]
        and row["changed_tensors"] > 0
        for row in attempts
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        record
        for record in json.loads(args.records.read_text())
        if record["task"] == "book-flight-nodelay"
    ]
    short = records[0]["steps"][0]["messages"]
    longest = records[1]["steps"][-1]["messages"]
    report = {"thresholds": {"max_logit_difference": MAX_LOGIT_DIFFERENCE}}

    eager = load("eager")
    baseline = forward(eager, [short, longest])
    release(eager)
    selected = load("sdpa")
    candidate = forward(selected, [short, longest])
    report["comparisons"] = []
    for before, after in zip(baseline, candidate, strict=True):
        difference = (before["logits"] - after["logits"]).abs()
        report["comparisons"].append(
            {
                "prompt_tokens": after["prompt_tokens"],
                "top1_agrees": bool(
                    before["logits"].argmax(-1).item()
                    == after["logits"].argmax(-1).item()
                ),
                "max_absolute_difference": difference.max().item(),
                "mean_absolute_difference": difference.mean().item(),
                "finite": before["finite"] and after["finite"],
                "eager_seconds": before["seconds"],
                "sdpa_seconds": after["seconds"],
                "eager_peak_vram_bytes": before["peak_vram_bytes"],
                "sdpa_peak_vram_bytes": after["peak_vram_bytes"],
            }
        )
    print(report["comparisons"], flush=True)
    sdpa_passed = all(
        row["finite"]
        and row["top1_agrees"]
        and row["max_absolute_difference"] < MAX_LOGIT_DIFFERENCE
        for row in report["comparisons"]
    )
    report["selected_attention"] = "sdpa" if sdpa_passed else "eager"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not sdpa_passed:
        release(selected)
        selected = load("eager")

    try:
        lora_stage(
            selected, report, longest, tree_source=records[1], output=args.output
        )
    except Exception as error:  # noqa: BLE001 - recorded; the diagnostic does not use LoRA.
        report["lora_stage_error"] = f"{type(error).__name__}: {error}"[:1000]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
