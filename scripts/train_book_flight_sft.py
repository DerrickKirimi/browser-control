"""Bounded native-call LoRA SFT pilot with paired development skill probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from preflight_lfm3_attention import LORA, load

from browser_control import book_flight_sft, book_flight_solver
from browser_control.book_flight_curriculum import (
    plan_examples,
    validate_initial_adapter,
)
from browser_control.book_flight_diagnostic import (
    DEVELOPMENT_SEEDS,
    SAMPLING_128,
    build_probes,
    score_probe,
    summarize_probes,
    summarize_rollouts,
)
from browser_control.book_flight_evaluation import collect_book_flight_episode
from browser_control.book_flight_ids import (
    augment_training_examples,
    compare_id_scores,
    validate_probe_pair,
)
from browser_control.book_flight_sft import encode_example, expert_examples
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets


def main() -> None:
    from peft import LoraConfig, PeftModel, get_peft_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--gradient-report", type=Path, required=True)
    parser.add_argument("--token-report", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--id-permutation-seed", type=int)
    parser.add_argument("--permuted-probes", type=Path)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--focus-skills", nargs="+")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--seen-training-fit", action="store_true")
    args = parser.parse_args()
    if args.updates <= 0 or args.accumulation <= 0:
        parser.error("updates and accumulation must be positive")
    if args.output_dir.exists():
        parser.error("Preserve existing evidence; choose a new output directory")
    gate = json.loads(args.gradient_report.read_text())
    if not gate["update_passed"] or gate["revision"] != MODEL_REVISION:
        parser.error("A passing finite-update gate for this pinned model is required")
    if gate["lora"] != LORA or gate["attention"] != "sdpa":
        parser.error("Finite-update gate used a different training configuration")
    corpus = json.loads(args.corpus.read_text())
    tokens = json.loads(args.token_report.read_text())
    encoder_path = Path(book_flight_sft.__file__)
    if (
        tokens["status"] != "passed"
        or tokens["revision"] != MODEL_REVISION
        or tokens["corpus_sha256"]
        != hashlib.sha256(args.corpus.read_bytes()).hexdigest()
        or tokens["encoder_sha256"]
        != hashlib.sha256(encoder_path.read_bytes()).hexdigest()
    ):
        parser.error(
            "Passing token validation must match the exact corpus and label encoder"
        )
    solver_hash = hashlib.sha256(
        Path(book_flight_solver.__file__).read_bytes()
    ).hexdigest()
    if corpus["status"] != "complete" or corpus["solver_sha256"] != solver_hash:
        parser.error("Corpus must be complete and use the current corrected solver")
    examples = [
        example for record in corpus["records"] for example in expert_examples(record)
    ]
    if examples != corpus["examples"]:
        parser.error("Corpus examples do not match the clean successful records")
    augmentation = None
    if args.id_permutation_seed is not None:
        examples, augmentation = augment_training_examples(
            examples, seed=args.id_permutation_seed
        )
    if tokens.get("id_augmentation") != augmentation:
        parser.error("Token validation must match the exact ID augmentation policy")
    if augmentation is not None and (args.focus_skills or args.passes != 1):
        parser.error("ID augmentation currently supports one full-corpus pass only")
    examples = plan_examples(examples, skills=args.focus_skills, passes=args.passes)
    needed = args.updates * args.accumulation
    if needed > len(examples):
        parser.error("Run exceeds the explicitly configured example/pass budget")
    run_label = (
        "adapter-continuation"
        if args.initial_adapter is not None
        else "id-randomized"
        if augmentation is not None
        else "multi-pass"
        if args.passes > 1
        else "corpus-pass"
        if len(examples) - needed < args.accumulation
        else "pilot"
    )
    probes = json.loads(args.probes.read_text())["probes"]
    if any(probe["task_seed"] not in DEVELOPMENT_SEEDS for probe in probes):
        parser.error("Skill probes must use development seeds only")
    permuted_probes = None
    if args.permuted_probes is not None:
        original = json.loads(args.probes.read_text())
        altered = json.loads(args.permuted_probes.read_text())
        validate_probe_pair(
            original,
            altered,
            original_sha256=hashlib.sha256(args.probes.read_bytes()).hexdigest(),
        )
        permuted_probes = altered["probes"]
    elif augmentation is not None:
        parser.error("ID-randomized SFT requires the matched permutation evaluation")
    initial_adapter = None
    if args.initial_adapter is not None:
        adapter_path = args.initial_adapter / "adapter_config.json"
        adapter_config = json.loads(adapter_path.read_text())
        validate_initial_adapter(
            adapter_config, LORA, revision=MODEL_REVISION, model_id=MODEL_ID
        )
        initial_adapter = {
            "config_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(
                (args.initial_adapter / "adapter_model.safetensors").read_bytes()
            ).hexdigest(),
        }
    source = Path("source-manifest.json")
    snapshot = json.loads(source.read_text()) if source.exists() else None
    if snapshot is not None:
        for name, expected in snapshot["files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")
    policy = load("sdpa")
    if args.initial_adapter is not None:
        policy.model = PeftModel.from_pretrained(
            policy.model, args.initial_adapter, is_trainable=True
        )
        policy.model.peft_config["default"].revision = MODEL_REVISION
    policy.max_prompt_tokens = 16384
    torch = policy.torch
    torch.manual_seed(23)
    encoded = [
        encode_example(policy.tokenizer, example) for example in examples[:needed]
    ]
    args.output_dir.mkdir(parents=True)
    report = {
        "status": "running",
        "source_snapshot": snapshot,
        "model_revision": MODEL_REVISION,
        "scope": f"Bounded {run_label} SFT; no held-out task estimate",
        "config": {
            "seed": 23,
            "run_label": run_label,
            "focus_skills": args.focus_skills,
            "passes": args.passes,
            "initial_adapter": initial_adapter,
            "optimizer_state": "fresh",
            "id_augmentation": augmentation,
            "updates": args.updates,
            "accumulation": args.accumulation,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "initial_scale": 1.0,
            "lora": LORA,
            "attention": "sdpa",
            "checkpointing": True,
            "labels": "current canonical native call through EOS only",
            "prefill_chunk_size": 512,
            "decoder": asdict(SAMPLING_128),
            "rollout_task_seeds": list(range(30, 40)),
            "rollout_max_steps": 15,
        },
        "inputs_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("corpus", args.corpus),
                ("gradient_report", args.gradient_report),
                ("token_report", args.token_report),
                ("probes", args.probes),
            )
        },
        "samples": [
            {
                k: example[k]
                for k in (
                    "task_seed",
                    "turn",
                    "skill",
                    "id_mapping_sha256",
                    "id_permutation_seed",
                    "pass_index",
                )
                if k in example
            }
            for example in examples[:needed]
        ],
        "prompt_tokens": [e["prompt_tokens"] for e in encoded],
        "target_tokens": [e["target_tokens"] for e in encoded],
        "updates": [],
        "example_losses": [],
        "skill_losses": {},
        "before": [],
        "after": [],
        "before_rollouts": [],
        "after_rollouts": [],
    }
    report["manifest"] = RunManifest(
        f"lfm3-book-flight-sft-{run_label}", 23, report["config"]
    ).to_dict()
    if args.permuted_probes is not None:
        report["inputs_sha256"]["permuted_probes"] = hashlib.sha256(
            args.permuted_probes.read_bytes()
        ).hexdigest()
        report["permuted_after"] = []
    if args.seen_training_fit:
        report["seen_training_after"] = []
    report_path = args.output_dir / "training.json"

    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(report_path)

    def evaluate(stage):
        policy.model.eval()
        policy.model.config.use_cache = True
        policy.model.generation_config.prefill_chunk_size = 512
        policy.decoder = SAMPLING_128
        for index, probe in enumerate(probes):
            seed = 100000 + probe["task_seed"] * 16 + index
            output = policy.generate(probe["messages"], seed=seed)
            report[stage].append(
                {
                    "task_seed": probe["task_seed"],
                    "skill": probe["skill"],
                    "prior_error": probe["prior_error"],
                    "generation_seed": seed,
                    **asdict(output),
                    **score_probe(probe, output.text, "strict"),
                }
            )
            save()
        report[f"{stage}_summary"] = summarize_probes(report[stage])
        save()
        root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
        with MiniwobAssetServer(root) as server:
            env = LocalBrowserGymEnvironment.miniwob(
                "book-flight-nodelay", miniwob_url=server.base_url
            )
            try:
                for seed in range(30, 40):
                    episode = collect_book_flight_episode(
                        env, policy, seed, numerical_check=policy.check_logits
                    )
                    report[f"{stage}_rollouts"].append(episode)
                    save()
                    print(stage, "rollout", seed, episode["success"], flush=True)
            finally:
                env.close()
        report[f"{stage}_rollout_summary"] = summarize_rollouts(
            report[f"{stage}_rollouts"]
        )
        save()

    save()
    try:
        evaluate("before")
        torch.manual_seed(23)
        model = (
            policy.model
            if args.initial_adapter is not None
            else get_peft_model(
                policy.model, LoraConfig(**LORA, revision=MODEL_REVISION)
            )
        )
        policy.model = model
        model.config.use_cache = False
        model.train()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not trainable or any("lora_" not in name for name in trainable):
            raise ValueError("Only LoRA parameters may be trainable")
        report["trainable_tensors"] = len(trainable)
        report["trainable_parameters"] = sum(p.numel() for p in trainable.values())
        if (
            report["trainable_tensors"] != gate["trainable_tensors"]
            or report["trainable_parameters"] != gate["trainable_parameters"]
        ):
            raise ValueError(
                "Trainable adapter shape differs from the finite-update gate"
            )
        report["adapter_dtypes"] = sorted({str(p.dtype) for p in trainable.values()})
        save()
        before = {name: p.detach().cpu().clone() for name, p in trainable.items()}
        optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
        torch.cuda.reset_peak_memory_stats()
        for update in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            started = time.monotonic()
            start = update * args.accumulation
            for position, example in enumerate(
                encoded[start : start + args.accumulation], start=start
            ):
                ids = torch.tensor([example["input_ids"]], device="cuda:0")
                target_length = example["target_tokens"]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        logits_to_keep=target_length + 1,
                    ).logits
                    loss = torch.nn.functional.cross_entropy(
                        logits[0, :-1].float(), ids[0, -target_length:]
                    )
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite loss; training stopped")
                losses.append(loss.item())
                metadata = examples[position]
                report["example_losses"].append(
                    {
                        "update": update + 1,
                        "task_seed": metadata["task_seed"],
                        "turn": metadata["turn"],
                        "pass_index": metadata["pass_index"],
                        "skill": metadata["skill"],
                        "loss": loss.item(),
                        "target_tokens": target_length,
                    }
                )
                skill = report["skill_losses"].setdefault(
                    metadata["skill"], {"count": 0, "sum": 0.0}
                )
                skill["count"] += 1
                skill["sum"] += loss.item()
                skill["mean"] = skill["sum"] / skill["count"]
                scaler.scale(loss / args.accumulation).backward()
                del logits, loss, ids
            scaler.unscale_(optimizer)
            bad = [
                name
                for name, p in trainable.items()
                if p.grad is None or not torch.isfinite(p.grad).all().item()
            ]
            if bad:
                report["bad_gradient_names"] = bad
                raise RuntimeError(
                    "Missing/nonfinite gradients; optimizer step refused"
                )
            norm = torch.nn.utils.clip_grad_norm_(
                trainable.values(), 1.0, error_if_nonfinite=True
            )
            if norm.item() <= 0:
                raise RuntimeError("Zero gradient; optimizer step refused")
            scaler.step(optimizer)
            scaler.update()
            if any(not torch.isfinite(p).all().item() for p in trainable.values()):
                raise RuntimeError("Nonfinite adapter weights after update")
            report["updates"].append(
                {
                    "update": update + 1,
                    "mean_loss": sum(losses) / len(losses),
                    "unclipped_gradient_norm": norm.item(),
                    "scale": scaler.get_scale(),
                    "seconds": time.monotonic() - started,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            save()
            print(report["updates"][-1], flush=True)
            if (update + 1) % 4 == 0 or update + 1 == args.updates:
                checkpoint = f"checkpoint-{update + 1}"
                model.save_pretrained(args.output_dir / checkpoint)
                report["last_checkpoint"] = checkpoint
                save()
        report["changed_adapter_tensors"] = sum(
            not torch.equal(p.detach().cpu(), before[name])
            for name, p in trainable.items()
        )
        if not report["changed_adapter_tensors"]:
            raise RuntimeError("No adapter weights changed")
        optimizer.zero_grad(set_to_none=True)
        del optimizer, before
        model.gradient_checkpointing_disable()
        torch.cuda.empty_cache()
        evaluate("after")
        if permuted_probes is not None:
            for index, probe in enumerate(permuted_probes):
                seed = 100000 + probe["task_seed"] * 16 + index
                output = policy.generate(probe["messages"], seed=seed)
                report["permuted_after"].append(
                    {
                        "task_seed": probe["task_seed"],
                        "skill": probe["skill"],
                        "prior_error": probe["prior_error"],
                        "generation_seed": seed,
                        **asdict(output),
                        **score_probe(probe, output.text, "strict"),
                    }
                )
                save()
            report["permuted_after_summary"] = summarize_probes(
                report["permuted_after"]
            )
            report["id_sensitivity_decision"] = compare_id_scores(
                report["after_summary"], report["permuted_after_summary"]
            )
        if args.seen_training_fit:
            seen = [
                probe
                for record in corpus["records"]
                for probe in build_probes(record)
                if args.focus_skills is None or probe["skill"] in args.focus_skills
            ]
            for index, probe in enumerate(seen):
                seed = 100000 + probe["task_seed"] * 16 + index
                output = policy.generate(probe["messages"], seed=seed)
                report["seen_training_after"].append(
                    {
                        "task_seed": probe["task_seed"],
                        "skill": probe["skill"],
                        "prior_error": probe["prior_error"],
                        "generation_seed": seed,
                        **asdict(output),
                        **score_probe(probe, output.text, "strict"),
                    }
                )
                save()
            report["seen_training_after_summary"] = summarize_probes(
                report["seen_training_after"]
            )
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
