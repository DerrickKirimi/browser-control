"""Group-relative policy gradient on multi-turn screenshot episodes, from a saved adapter.

For each training task, sample a group of episodes at the evaluation decoder, score each by the page's
terminal reward, and update once with every turn's tokens weighted by its episode's advantage. Groups
with equal rewards produce no update and are recorded as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from train_lfm3_vision_grpo import LORA_TARGETS, frozen_weights_hash


def generation_seed(task_seed: int, member: int) -> int:
    return 3_000_000 + task_seed * 64 + member * 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="click-button-sequence")
    parser.add_argument("--task-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Preserve earlier training evidence")
    if any(not 100 <= s < 10000 for s in args.task_seeds) or len(set(args.task_seeds)) != len(args.task_seeds):
        parser.error("Use distinct training seeds 100..9999")
    if not 2 <= args.group_size <= 8 or not 1 <= args.max_steps <= 8:
        parser.error("Group size 2..8 and max steps 1..8")

    import torch
    from peft import PeftModel

    from browser_control.lfm3_optimization import checked_lora_step
    from browser_control.lfm3_vision import (
        MODEL_ID,
        MODEL_REVISION,
        LFM3VisionPolicy,
        adapter_hash,
    )
    from browser_control.manifest import RunManifest
    from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
    from browser_control.transformers_policy import DecoderConfig
    from browser_control.vision_browser import VisionBrowserGymEnvironment
    from browser_control.vision_sequence_grpo import (
        group_advantages,
        sample_episode,
        training_turns,
    )

    updates = len(args.task_seeds)
    decoder = DecoderConfig("sampling", True, 128, temperature=1.0, top_p=0.95)
    args.output_dir.mkdir(parents=True)
    report = {
        "status": "running",
        "scope": "group-relative policy gradient on multi-turn screenshot episodes; matched evaluation is separate",
        "groups": [],
        "manifest": RunManifest(
            "lfm3-vision-sequence-grpo",
            args.training_seed,
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "task": args.task,
                "task_seeds": args.task_seeds,
                "group_size": args.group_size,
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "lr_schedule": "linear to zero over one update slot per task",
                "decoder": {"name": "sampling", "temperature": 1.0, "top_p": 0.95, "max_new_tokens": 128},
                "generation_seed": "3000000 + task_seed * 64 + member * 16 + turn",
                "advantage": "episode reward minus group mean; no std normalisation; equal groups skipped",
                "loss": "-advantage * sum of completion-token log-probabilities over the episode's turns / group tokens",
                "init_adapter_sha256": adapter_hash(args.init_adapter),
                "lora": {"r": 32, "lora_alpha": 64, "targets": LORA_TARGETS},
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
        policy = LFM3VisionPolicy.load(decoder)
        torch.manual_seed(args.training_seed)
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
        optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / updates))
        scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
        torch.cuda.reset_peak_memory_stats()
        root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
        with MiniwobAssetServer(root) as server, closing(
            VisionBrowserGymEnvironment.miniwob(args.task, miniwob_url=server.base_url)
        ) as env:
            for index, task_seed in enumerate(args.task_seeds):
                model.eval()
                episodes = [
                    sample_episode(env, policy, task_seed, generation_seed(task_seed, member), args.max_steps)
                    for member in range(args.group_size)
                ]
                rewards = [episode.reward if episode.success else 0.0 for episode in episodes]
                advantages = group_advantages(rewards)
                turns, tokens = training_turns(episodes, advantages)
                record = {
                    "task_seed": task_seed,
                    "rewards": rewards,
                    "advantages": advantages,
                    "turns": [len(e.turns) for e in episodes],
                    "valid": [all(t.valid for t in e.turns) for e in episodes],
                    "texts": [[t.text for t in e.turns] for e in episodes],
                    "group_tokens": tokens,
                    "varied": bool(turns),
                }
                if turns:
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    objective = 0.0
                    for turn, advantage in turns:
                        inputs = policy.encode(turn.observation)
                        completion = torch.tensor(turn.completion_token_ids, device=inputs["input_ids"].device)
                        ids = torch.cat([inputs["input_ids"][0], completion]).unsqueeze(0)
                        forward = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
                        length = len(completion)
                        with torch.autocast("cuda", dtype=torch.float16):
                            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                           logits_to_keep=length + 1, **forward).logits
                        logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
                        token_logprobs = logprobs.gather(1, completion.unsqueeze(1)).squeeze(1)
                        loss = -advantage * token_logprobs.sum() / tokens
                        if not torch.isfinite(loss).item():
                            raise RuntimeError("Nonfinite loss; training stopped")
                        objective += loss.item()
                        scaler.scale(loss).backward()
                    step = checked_lora_step(trainable, optimizer, scaler)
                    if not step["nonzero_gradient"]:
                        raise RuntimeError("A group with varied rewards produced a zero gradient")
                    record.update(loss=objective, **step)
                scheduler.step()
                report["groups"].append(record)
                if index % 10 == 0:
                    save()
        changed = sum(not torch.equal(p.detach().cpu(), before[n]) for n, p in trainable.items())
        varied = sum(g["varied"] for g in report["groups"])
        report.update(
            varied_groups=varied,
            optimizer_steps=varied,
            changed_adapter_tensors=changed,
            trainable_tensors=len(trainable),
            frozen_after_sha256=frozen_weights_hash(model),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        model.save_pretrained(args.output_dir / "adapter")
        report["adapter_sha256"] = hashlib.sha256((args.output_dir / "adapter/adapter_model.safetensors").read_bytes()).hexdigest()
        if not varied:
            raise ValueError("No group had varied rewards; nothing was trained")
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
