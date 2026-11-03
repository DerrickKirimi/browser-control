"""Native-reward GRPO on pretrained LFM3-VL-2B, on one-click screenshot tasks.

At most four updates form an engineering contract. A longer run needs that passed contract and a separate matched
evaluation. Only language-model LoRA trains; the vision tower and the base stay frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

LORA_TARGETS = r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)$"
MAX_UPDATES = 300


def frozen_weights_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(f"{name}:{parameter.dtype}:{list(parameter.shape)}".encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-report", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    args = parser.parse_args()
    updates = len(args.task_seeds)
    contract = updates <= 4
    if (
        updates > MAX_UPDATES
        or len(set(args.task_seeds)) != updates
        or any(not 100 <= seed < 10000 for seed in args.task_seeds)
        or args.group_size not in (2, 4, 8)
        or not 0 < args.learning_rate <= 1e-3
    ):
        parser.error("Declare distinct training tasks 100..9999, groups of 2/4/8, and a bounded rate")
    if not contract and args.contract_report is None:
        parser.error("A learning run requires a passed real contract")
    if args.output_dir.exists():
        parser.error("Preserve earlier training evidence")
    if not contract:
        passed = json.loads(args.contract_report.read_text())
        if passed["status"] != "passed" or passed["group_size"] != args.group_size:
            parser.error("The contract did not pass, or differs in group size")

    import torch
    import trl
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import TrainerCallback

    from browser_control.book_flight_rollout import native_success_reward
    from browser_control.lfm3_vision import MODEL_ID, MODEL_REVISION, LFM3VisionPolicy
    from browser_control.manifest import RunManifest
    from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
    from browser_control.transformers_policy import DecoderConfig
    from browser_control.vision_browser import VisionBrowserGymEnvironment
    from browser_control.vision_rollout import VisionGroupCollector
    from browser_control.vision_trainer import (
        StagedVisionRollout,
        native_vision_grpo_trainer_class,
    )

    args.output_dir.mkdir(parents=True)
    decoder = DecoderConfig("vision-grpo", True, 128, temperature=1.0, top_p=0.95)
    report = {
        "status": "running",
        "scope": "engineering contract, not a learning result" if contract else "learning run; matched evaluation is separate",
        "group_size": args.group_size,
        "task": args.task,
        "task_seeds": args.task_seeds,
        "groups": [],
        "loss_checks": [],
        "gradient_checks": [],
        "manifest": RunManifest(
            "lfm3-vision-grpo",
            23,
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "task": args.task,
                "task_seeds": args.task_seeds,
                "group_size": args.group_size,
                "learning_rate": args.learning_rate,
                "decoder": {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 128},
                "lora": {"r": 32, "lora_alpha": 64, "targets": LORA_TARGETS},
                "generation_seed": "2000000 + task_seed * 64 + member * 16",
            },
        ).to_dict(),
        "runtime_versions": {
            name: version(name) for name in ("torch", "torchvision", "transformers", "trl", "peft", "accelerate")
        },
    }
    snapshot = Path("source-manifest.json")
    if snapshot.exists():
        report["source_snapshot"] = json.loads(snapshot.read_text())
        for name, expected in report["source_snapshot"]["files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError("Source hash differs: " + name)

    def save():
        temporary = args.output_dir / "training.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output_dir / "training.json")

    save()
    try:
        policy = LFM3VisionPolicy.load(decoder)
        torch.manual_seed(23)
        # PEFT injects LoRA layers in place, so policy.model samples with the current adapter.
        model = get_peft_model(
            policy.model,
            LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"),
        )
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if not trainable or any("lora_" not in n or "language_model" not in n for n in trainable):
            raise ValueError("Only language-model LoRA may train")
        if any(p.dtype != torch.float32 for p in trainable.values()):
            raise ValueError("The FP16 base requires float32 adapters")
        report["trainable_tensors"] = len(trainable)
        before = {name: p.detach().cpu().clone() for name, p in trainable.items()}
        report["frozen_before_sha256"] = frozen_weights_hash(model)
        save()
        root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
        with (
            MiniwobAssetServer(root) as server,
            closing(VisionBrowserGymEnvironment.miniwob(args.task, miniwob_url=server.base_url)) as env,
        ):
            collector = VisionGroupCollector(env, policy, args.output_dir / "screenshots")
            rollout = StagedVisionRollout(collector)

            class Checks(TrainerCallback):
                def on_pre_optimizer_step(self, trainer_args, state, control, **kwargs):
                    gradients = [p.grad for p in trainable.values() if p.grad is not None]
                    if len(gradients) != len(trainable) or any(not torch.isfinite(g).all() for g in gradients):
                        raise ValueError("Missing or nonfinite adapter gradients")
                    norm = torch.linalg.vector_norm(torch.stack([g.float().norm() for g in gradients])).item()
                    varied = collector.groups[-1]["reward_variation"]
                    if varied and norm == 0:
                        raise ValueError("A group with varied reward produced no gradient")
                    if not varied and norm != 0:
                        raise ValueError("A group with equal rewards produced a gradient")
                    report["gradient_checks"].append({"step": state.global_step, "norm": norm, "varied": varied})

                def on_step_end(self, trainer_args, state, control, **kwargs):
                    if any(not torch.isfinite(p).all() for p in trainable.values()):
                        raise ValueError("Nonfinite adapter after an optimizer step")
                    report["optimizer_steps"] = state.global_step
                    report["groups"] = collector.groups
                    report["episodes"] = collector.episodes
                    report["loss_checks"] = trainer.vision_loss_checks
                    if args.checkpoint_every and state.global_step % args.checkpoint_every == 0:
                        model.save_pretrained(args.output_dir / f"checkpoint-{state.global_step}")
                    save()

            config = trl.GRPOConfig(
                output_dir=str(args.output_dir / "trainer"),
                fp16=True,
                bf16=False,
                use_vllm=False,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=args.group_size,
                num_generations=args.group_size,
                steps_per_generation=args.group_size,
                num_iterations=1,
                max_steps=updates,
                max_completion_length=128,
                learning_rate=args.learning_rate,
                weight_decay=0.0,
                max_grad_norm=1.0,
                beta=0.0,
                loss_type="grpo",
                scale_rewards="group",
                mask_truncated_completions=False,
                disable_dropout=True,
                gradient_checkpointing=False,
                temperature=1.0,
                top_p=0.95,
                report_to="none",
                save_strategy="no",
                logging_steps=1,
                seed=23,
            )
            trainer = native_vision_grpo_trainer_class()(
                model=model,
                args=config,
                processing_class=policy.processor,
                reward_funcs=native_success_reward,
                train_dataset=Dataset.from_list(
                    [{"prompt": json.dumps({"task_seed": seed})} for seed in args.task_seeds]
                ),
                rollout_func=rollout,
                callbacks=[Checks()],
            )
            if trainer.accelerator.scaler is None or trainer.accelerator.scaler.get_scale() != 1.0:
                raise ValueError("The FP16 scaler must start at 1")
            torch.cuda.reset_peak_memory_stats()
            trainer.train()
        changed = sum(not torch.equal(p.detach().cpu(), before[n]) for n, p in trainable.items())
        varied = sum(group["reward_variation"] for group in collector.groups)
        report.update(
            changed_adapter_tensors=changed,
            varied_groups=varied,
            frozen_after_sha256=frozen_weights_hash(model),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            final_scale=trainer.accelerator.scaler.get_scale(),
            binding_checks=len(trainer.vision_binding_checks),
        )
        model.save_pretrained(args.output_dir / "adapter")
        problems = []
        if report["frozen_before_sha256"] != report["frozen_after_sha256"]:
            problems.append("frozen weights changed")
        if report["optimizer_steps"] != updates:
            problems.append("wrong number of updates")
        if report["binding_checks"] != updates * args.group_size:
            problems.append("not every microbatch was bound to its sampled row")
        if varied and changed != len(trainable):
            problems.append("not every adapter tensor changed")
        if contract and not varied:
            problems.append("no group varied, so the contract tested nothing")
        if not all(row.get("external_gradients_zero") for row in trainer.vision_loss_checks):
            problems.append("gradient mask check missing")
        report["status"] = "failed" if problems else ("passed" if contract else "complete")
        report["problems"] = problems
        save()
        if problems:
            raise ValueError("; ".join(problems))
    except BaseException as error:
        if report["status"] == "running":
            report.update(status="failed", error=f"{type(error).__name__}: {error}"[:2000])
            save()
        raise


if __name__ == "__main__":
    main()
