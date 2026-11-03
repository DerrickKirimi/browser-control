"""Supervised control for vision GRPO: the same tasks, sample budget, LoRA and optimizer, with expert labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

from train_lfm3_vision_grpo import LORA_TARGETS, frozen_weights_hash


def canonical_call(x: int, y: int) -> str:
    return (
        f"<tool_call>\n<function=click>\n<parameter=x>\n{x}\n</parameter>\n"
        f"<parameter=y>\n{y}\n</parameter>\n</function>\n</tool_call>"
    )


def schedule(examples: list[dict], passes: int, seed: int) -> list[dict]:
    """Each task appears once per pass, as each GRPO task yields one sample per group member."""
    generator = random.Random(seed)
    order = []
    for _ in range(passes):
        shuffled = list(examples)
        generator.shuffle(shuffled)
        order.extend(shuffled)
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--training-seed", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--init-adapter", type=Path, help="Continue from this LoRA adapter instead of a fresh one"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Preserve earlier training evidence")

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    from browser_control.lfm3_optimization import checked_lora_step
    from browser_control.lfm3_vision import (
        MODEL_ID,
        MODEL_REVISION,
        LFM3VisionPolicy,
        adapter_hash,
    )
    from browser_control.manifest import RunManifest
    from browser_control.transformers_policy import DecoderConfig
    from browser_control.vision_browser import VisionObservation, parse_coordinate_click

    corpus = json.loads(args.corpus.read_text())
    examples = corpus["examples"]
    order = schedule(examples, args.passes, args.training_seed)
    updates = len(order) // args.accumulation
    if not updates or len(order) % args.accumulation:
        parser.error("The example count must divide into whole updates")
    args.output_dir.mkdir(parents=True)
    report = {
        "status": "running",
        "scope": "supervised vision control from expert labels; matched evaluation is separate",
        "inputs_sha256": {"corpus": hashlib.sha256(args.corpus.read_bytes()).hexdigest()},
        "updates": [],
        "manifest": RunManifest(
            "lfm3-vision-sft",
            args.training_seed,
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "learning_rate": args.learning_rate,
                "examples": len(order),
                "accumulation": args.accumulation,
                "planned_updates": updates,
                "lr_schedule": "linear to zero, matching the GRPO trainer default",
                "lora": {"r": 32, "lora_alpha": 64, "targets": LORA_TARGETS},
                "init_adapter_sha256": adapter_hash(args.init_adapter),
                "corpus_task": corpus.get("task", "click-color"),
            },
        ).to_dict(),
        "runtime_versions": {name: version(name) for name in ("torch", "transformers", "peft")},
    }

    def save():
        temporary = args.output_dir / "training.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output_dir / "training.json")

    save()
    try:
        policy = LFM3VisionPolicy.load(DecoderConfig("greedy", False, 128))
        torch.manual_seed(args.training_seed)
        if args.init_adapter is None:
            model = get_peft_model(
                policy.model,
                LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"),
            )
        else:
            model = PeftModel.from_pretrained(policy.model, args.init_adapter, is_trainable=True)
            config = model.peft_config["default"]
            if (config.r, config.lora_alpha, config.target_modules) != (32, 64, LORA_TARGETS):
                raise ValueError("The starting adapter differs from the declared LoRA shape")
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if not trainable or any("lora_" not in n or "language_model" not in n for n in trainable):
            raise ValueError("Only language-model LoRA may train")
        if any(p.dtype != torch.float32 for p in trainable.values()):
            raise ValueError("LoRA weights must train in float32")
        before = {name: p.detach().cpu().clone() for name, p in trainable.items()}
        report["frozen_before_sha256"] = frozen_weights_hash(model)
        tokenizer = policy.processor.tokenizer
        images = args.corpus.parent / "expert-screenshots"
        encoded = {}
        for example in examples:
            data = (images / example["screenshot"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != example["screenshot_sha256"]:
                raise ValueError("Expert screenshot differs")
            target = example["target"]
            text = canonical_call(target["x"], target["y"])
            parsed = parse_coordinate_click(text)
            if (parsed.x, parsed.y) != (target["x"], target["y"]):
                raise ValueError("The canonical call does not parse back to its target")
            observation = VisionObservation(example["goal"], data, example["width"], example["height"])
            inputs = policy.encode(observation)
            labels = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            encoded[example.get("example_id", example["task_seed"])] = (inputs, torch.tensor(labels, device="cuda:0"))
        model.train()
        optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / updates))
        scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
        torch.cuda.reset_peak_memory_stats()
        for update in range(updates):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for example in order[update * args.accumulation : (update + 1) * args.accumulation]:
                inputs, labels = encoded[example.get("example_id", example["task_seed"])]
                ids = torch.cat([inputs["input_ids"][0], labels]).unsqueeze(0)
                forward = {key: value for key, value in inputs.items() if key not in ("input_ids", "attention_mask")}
                length = len(labels)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=length + 1, **forward).logits
                    loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, -length:])
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite loss; training stopped")
                losses.append(loss.item())
                scaler.scale(loss / args.accumulation).backward()
            step = checked_lora_step(trainable, optimizer, scaler)
            scheduler.step()
            report["updates"].append({"update": update + 1, "mean_loss": sum(losses) / len(losses), **step})
            if update % 10 == 0:
                save()
        changed = sum(not torch.equal(p.detach().cpu(), before[n]) for n, p in trainable.items())
        report.update(
            changed_adapter_tensors=changed,
            trainable_tensors=len(trainable),
            frozen_after_sha256=frozen_weights_hash(model),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        model.save_pretrained(args.output_dir / "adapter")
        if changed != len(trainable) or report["frozen_before_sha256"] != report["frozen_after_sha256"]:
            raise ValueError("Adapter or frozen-weight check failed")
        report["status"] = "complete"
        save()
    except BaseException as error:
        if report["status"] == "running":
            report.update(status="failed", error=f"{type(error).__name__}: {error}"[:2000])
            save()
        raise


if __name__ == "__main__":
    main()
