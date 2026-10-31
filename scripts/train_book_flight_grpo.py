"""Bounded pretrained LFM3/browser GRPO after frozen token-group checks.

The default single update is an engineering smoke. Longer learning runs require
its passed contract, explicit training tasks and separate matched evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from types import MethodType

from check_book_flight_token_rollouts import audit_history
from preflight_lfm3_attention import LORA

from browser_control.book_flight_curriculum import validate_initial_adapter
from browser_control.book_flight_rollout import (
    BookFlightRollout,
    TransformersTokenSampler,
    native_success_reward,
    validate_trainer_token_rows,
)
from browser_control.book_flight_sft import TRAINING_SEEDS
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION, LFM3Policy
from browser_control.lfm3_trainer import low_scale_grpo_trainer_class
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.transformers_policy import DecoderConfig


def frozen_weights_hash(model):
    """Hash frozen tensors sequentially, without retaining a second full model."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(f"{name}:{parameter.dtype}:{list(parameter.shape)}".encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    import torch
    import trl
    from datasets import Dataset
    from peft import PeftModel
    from transformers import TrainerCallback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--rollout-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--task-seeds", type=int, nargs="+")
    parser.add_argument("--contract-report", type=Path)
    args = parser.parse_args()
    learning = args.updates > 1
    if not 1 <= args.updates <= 8 or not 0 < args.learning_rate <= 1e-5:
        parser.error("Declare 1..8 updates and learning rate in (0, 1e-5]")
    if learning and (args.contract_report is None or not args.task_seeds):
        parser.error(
            "Learning requires a passed real update contract and explicit tasks"
        )
    if not learning and args.task_seeds is not None:
        parser.error("The single-step contract repeats the frozen diagnostic task")
    if args.output_dir.exists():
        parser.error("Preserve earlier training evidence")
    diagnostic = json.loads(args.rollout_report.read_text())
    weights_hash = hashlib.sha256(
        (args.adapter / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    if (
        diagnostic["status"] != "complete"
        or not diagnostic["canonical_history_ids_match"]
        or diagnostic["manifest"]["config"]["adapter_sha256"] != weights_hash
        or diagnostic["manifest"]["config"]["model_revision"] != MODEL_REVISION
    ):
        parser.error("Completed matched-adapter canonical token diagnostic required")
    groups = [g for g in diagnostic["groups"] if g["reward_variation"]]
    if not groups:
        parser.error("Frozen diagnostic found no native within-group reward variation")
    seed = groups[0]["task_seed"]
    if type(seed) is not int or seed not in TRAINING_SEEDS:
        parser.error("Only training seeds 100..9999 are allowed")
    task_seeds = args.task_seeds if learning else [seed]
    if (
        len(task_seeds) > 8
        or len(set(task_seeds)) != len(task_seeds)
        or len(task_seeds) < args.updates
        or any(seed not in TRAINING_SEEDS for seed in task_seeds)
    ):
        parser.error("Declare distinct training tasks, at least one per planned update")
    validate_initial_adapter(
        json.loads((args.adapter / "adapter_config.json").read_text()),
        LORA,
        revision=MODEL_REVISION,
        model_id=MODEL_ID,
    )
    settings = diagnostic["manifest"]["config"]["decoder"]
    decoder = DecoderConfig(**{**settings, "name": "grpo-smoke"})
    group_size = diagnostic["manifest"]["config"]["group_size"]
    if (
        group_size not in (2, 4)
        or decoder.generation_kwargs()
        != {
            "do_sample": True,
            "max_new_tokens": 128,
            "temperature": decoder.temperature,
            "top_p": 0.95,
        }
        or decoder.temperature not in (1.0, 1.5)
    ):
        parser.error(
            "Bounded smoke requires matched groups2/4, temperature1/1.5, top_p.95, cap128"
        )
    if learning:
        contract = json.loads(args.contract_report.read_text())
        if (
            contract["status"] != "passed"
            or contract["optimizer_steps"] != 1
            or not contract["changed_adapter_tensors"]
            or contract["adapter_sha256"] != weights_hash
            or contract["model_revision"] != MODEL_REVISION
            or contract["frozen_before_sha256"] != contract["frozen_after_sha256"]
            or contract["manifest"]["config"]["num_generations"] != group_size
            or DecoderConfig(
                **contract["manifest"]["config"]["decoder"]
            ).generation_kwargs()
            != decoder.generation_kwargs()
            or not all(
                row.get("raw_ids_preserved") and row.get("external_gradients_zero")
                for row in contract["loss_checks"]
            )
        ):
            parser.error(
                "Real contract must match original SFT weights, decoder and group size"
            )
    args.output_dir.mkdir(parents=True)
    report = {
        "status": "running",
        "scope": "bounded native-reward learning attempt; matched evaluation separate"
        if learning
        else "one engineering update; not an SFT versus GRPO learning comparison",
        "model_revision": MODEL_REVISION,
        "adapter_sha256": weights_hash,
        "rollout_report_sha256": hashlib.sha256(
            args.rollout_report.read_bytes()
        ).hexdigest(),
        "task_seed": seed,
        "task_seeds": task_seeds,
        "groups": [],
        "episodes": [],
        "loss_checks": [],
        "gradient_checks": [],
        "manifest": RunManifest(
            "lfm3-book-flight-grpo-learning"
            if learning
            else "lfm3-book-flight-grpo-smoke",
            23,
            {
                "decoder": asdict(decoder),
                "optimizer_steps": args.updates,
                "learning_rate": args.learning_rate,
                "microbatch_size": 1,
                "gradient_accumulation_steps": group_size,
                "num_generations": group_size,
                "initial_scale": 1.0,
            },
        ).to_dict(),
    }
    if args.contract_report is not None:
        report["contract_report_sha256"] = hashlib.sha256(
            args.contract_report.read_bytes()
        ).hexdigest()
    snapshot = Path("source-manifest.json")
    if snapshot.exists():
        report["source_snapshot"] = json.loads(snapshot.read_text())
        for name, expected in report["source_snapshot"]["files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")

    def save():
        temporary = args.output_dir / "training.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output_dir / "training.json")

    save()
    try:
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise ValueError("This bounded FP16 smoke requires a T4")
        policy = LFM3Policy.load(decoder, "native-tools", "sdpa")
        model = PeftModel.from_pretrained(policy.model, args.adapter, is_trainable=True)
        if model.get_input_embeddings().weight.dtype != torch.float16:
            raise ValueError("Expected the pinned FP16 base")
        model.generation_config.prefill_chunk_size = 512
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if len(trainable) != 284 or any("lora_" not in n for n in trainable):
            raise ValueError("Expected exactly 284 trainable LoRA tensors")
        if any(parameter.dtype != torch.float32 for parameter in trainable.values()):
            raise ValueError("FP16 base requires float32 trainable adapters")
        before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        frozen_before = frozen_weights_hash(model)
        root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
        with (
            MiniwobAssetServer(root) as server,
            closing(
                LocalBrowserGymEnvironment.miniwob(
                    "book-flight-nodelay", miniwob_url=server.base_url
                )
            ) as env,
        ):
            rollout = BookFlightRollout(
                env,
                policy.tokenizer,
                lambda current: TransformersTokenSampler(
                    current, policy.tokenizer, torch, decoder
                ),
                generation_seed_base=1_000_000 if learning else 1_000_000 + seed * 64,
            )

            def guarded_rollout(prompts, trainer):
                previous = len(rollout.episodes)
                output = rollout(prompts, trainer)
                report["episodes"] = rollout.episodes
                for episode in rollout.episodes[previous:]:
                    episode["history_audit"] = audit_history(episode, policy.tokenizer)
                report["groups"].append(
                    {
                        "optimizer_step": trainer.state.global_step,
                        "task_seed": rollout.episodes[previous]["task_seed"],
                        "native_rewards": [
                            float(value) for value in output["native_success"]
                        ],
                        "reward_variation": len(set(output["native_success"])) == 2,
                    }
                )
                save()
                if any(
                    not row["context_ids_match"]
                    or not row.get("assistant_ids_match", True)
                    for episode in rollout.episodes
                    for row in episode["history_audit"]
                ):
                    raise ValueError(
                        "Raw histories differ from the canonical evaluation contract"
                    )
                if not learning and len(set(output["native_success"])) != 2:
                    raise ValueError(
                        "Actual training group has no native reward variation"
                    )
                return output

            class CheckGradients(TrainerCallback):
                def on_pre_optimizer_step(self, args, state, control, **kwargs):
                    gradients = [
                        p.grad for p in trainable.values() if p.grad is not None
                    ]
                    if len(gradients) != len(trainable) or any(
                        not torch.isfinite(g).all() for g in gradients
                    ):
                        raise ValueError("Missing/nonfinite adapter gradients")
                    norm = torch.linalg.vector_norm(
                        torch.stack([g.float().norm() for g in gradients])
                    ).item()
                    if not 0 <= norm < float("inf") or (
                        report["groups"][-1]["reward_variation"] and norm == 0
                    ):
                        raise ValueError(
                            "Adapter gradient norm must be finite and positive"
                        )
                    report["gradient_checks"].append(
                        {"step": state.global_step, "norm": norm}
                    )
                    save()

                def on_step_end(self, args, state, control, **kwargs):
                    if any(not torch.isfinite(p).all() for p in trainable.values()):
                        raise ValueError("Nonfinite adapter after optimizer step")
                    report["optimizer_steps"] = state.global_step
                    if learning and (
                        state.global_step % 4 == 0
                        or state.global_step == planned_updates
                    ):
                        checkpoint = f"checkpoint-{state.global_step}"
                        model.save_pretrained(output_directory / checkpoint)
                        report["last_checkpoint"] = checkpoint
                    save()

            planned_updates = args.updates
            output_directory = args.output_dir
            config = trl.GRPOConfig(
                output_dir=str(args.output_dir / "trainer"),
                fp16=True,
                bf16=False,
                use_vllm=False,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=group_size,
                num_generations=group_size,
                steps_per_generation=group_size,
                num_iterations=1,
                max_steps=args.updates,
                max_completion_length=15 * 128,
                learning_rate=args.learning_rate,
                weight_decay=0.0,
                max_grad_norm=1.0,
                beta=0.0,
                loss_type="grpo",
                scale_rewards="group",
                mask_truncated_completions=False,
                disable_dropout=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                temperature=decoder.temperature,
                top_p=decoder.top_p,
                report_to="none",
                save_strategy="no",
                logging_steps=1,
                seed=23,
            )
            trainer = low_scale_grpo_trainer_class()(
                model=model,
                args=config,
                processing_class=policy.tokenizer,
                reward_funcs=native_success_reward,
                train_dataset=Dataset.from_list(
                    [{"prompt": json.dumps({"task_seed": seed})} for seed in task_seeds]
                ),
                rollout_func=guarded_rollout,
                callbacks=[CheckGradients()],
            )
            if (
                trainer.accelerator.scaler is None
                or trainer.accelerator.scaler.get_scale() != 1.0
            ):
                raise ValueError("Real FP16 scaler must start at 1")
            original_loss = trainer._compute_loss
            original_logps = trainer._get_per_token_logps_and_entropies

            def checked_loss(self, current_model, inputs):
                validate_trainer_token_rows(inputs, rollout.episodes)
                self._smoke_mask = inputs["completion_mask"] * inputs["tool_mask"]
                nonzero = bool(torch.any(inputs["advantages"] != 0).item())
                if not torch.isfinite(inputs["advantages"]).all() or (
                    not learning and not nonzero
                ):
                    raise ValueError(
                        "Native group must yield finite nonzero advantages"
                    )
                report["loss_checks"].append(
                    {
                        "advantages": inputs["advantages"].tolist(),
                        "raw_ids_preserved": True,
                        "nonzero_advantage": nonzero,
                    }
                )
                return original_loss(current_model, inputs)

            def checked_logps(self, *positional, **kwargs):
                result = original_logps(*positional, **kwargs)
                if result[0].requires_grad:
                    active = self._smoke_mask
                    row = report["loss_checks"][-1]

                    def check_gradient(gradient):
                        if not torch.isfinite(gradient).all() or torch.any(
                            gradient[active == 0] != 0
                        ):
                            raise ValueError(
                                "Nonfinite or external-token policy gradient"
                            )
                        nonzero = bool(torch.any(gradient[active == 1] != 0).item())
                        if nonzero != row["nonzero_advantage"]:
                            raise ValueError(
                                "No generated token received policy gradient"
                            )
                        row["external_gradients_zero"] = True

                    result[0].register_hook(check_gradient)
                return result

            trainer._compute_loss = MethodType(checked_loss, trainer)
            trainer._get_per_token_logps_and_entropies = MethodType(
                checked_logps, trainer
            )
            torch.cuda.reset_peak_memory_stats()
            trainer.train()
            changed = sum(
                not torch.equal(p.detach().cpu(), before[n])
                for n, p in trainable.items()
            )
            frozen_after = frozen_weights_hash(model)
            if (
                trainer.state.global_step != args.updates
                or (
                    any(row["reward_variation"] for row in report["groups"])
                    and not changed
                )
                or frozen_after != frozen_before
                or len(report["loss_checks"]) != group_size * args.updates
                or len(report["gradient_checks"]) != args.updates
                or len(rollout.episodes) != group_size * args.updates
                or not all(
                    row.get("external_gradients_zero") and row.get("raw_ids_preserved")
                    for row in report["loss_checks"]
                )
                or any(not torch.isfinite(p).all() for p in trainable.values())
            ):
                raise ValueError(
                    "Real update failed weights, masks, gradients or step checks"
                )
            model.save_pretrained(args.output_dir / "adapter")
            report.update(
                status="complete" if learning else "passed",
                optimizer_steps=trainer.state.global_step,
                changed_adapter_tensors=changed,
                frozen_before_sha256=frozen_before,
                frozen_after_sha256=frozen_after,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                final_scale=trainer.accelerator.scaler.get_scale(),
                useful_native_groups=sum(
                    row["reward_variation"] for row in report["groups"]
                ),
                nonzero_gradient_steps=sum(
                    row["norm"] > 0 for row in report["gradient_checks"]
                ),
            )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
