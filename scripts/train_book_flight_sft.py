"""Bounded native-call LoRA SFT pilot with paired development skill probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from preflight_lfm3_attention import LORA, load

from browser_control import book_flight_sft, book_flight_solver
from browser_control.book_flight_diagnostic import (
    DEVELOPMENT_SEEDS,
    SAMPLING_128,
    score_probe,
    summarize_probes,
    summarize_rollouts,
)
from browser_control.book_flight_evaluation import collect_book_flight_episode
from browser_control.book_flight_sft import encode_example, expert_examples
from browser_control.lfm3 import MODEL_REVISION
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets


def main() -> None:
    from peft import LoraConfig, get_peft_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--gradient-report", type=Path, required=True)
    parser.add_argument("--token-report", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
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
    needed = args.updates * args.accumulation
    if needed > len(examples):
        parser.error("Pilot must not repeat examples; collect more or reduce updates")
    probes = json.loads(args.probes.read_text())["probes"]
    if any(probe["task_seed"] not in DEVELOPMENT_SEEDS for probe in probes):
        parser.error("Skill probes must use development seeds only")
    source = Path("source-manifest.json")
    snapshot = json.loads(source.read_text()) if source.exists() else None
    if snapshot is not None:
        for name, expected in snapshot["files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")
    policy = load("sdpa")
    policy.max_prompt_tokens = 16384
    torch = policy.torch
    torch.manual_seed(23)
    random.Random(23).shuffle(examples)
    encoded = [
        encode_example(policy.tokenizer, example) for example in examples[:needed]
    ]
    args.output_dir.mkdir(parents=True)
    report = {
        "status": "running",
        "source_snapshot": snapshot,
        "model_revision": MODEL_REVISION,
        "scope": "SFT pilot; no held-out task estimate",
        "config": {
            "seed": 23,
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
            {k: example[k] for k in ("task_seed", "turn", "skill")}
            for example in examples[:needed]
        ],
        "prompt_tokens": [e["prompt_tokens"] for e in encoded],
        "target_tokens": [e["target_tokens"] for e in encoded],
        "updates": [],
        "before": [],
        "after": [],
        "before_rollouts": [],
        "after_rollouts": [],
    }
    report["manifest"] = RunManifest(
        "lfm3-book-flight-sft-pilot", 23, report["config"]
    ).to_dict()
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
        model = get_peft_model(policy.model, LoraConfig(**LORA))
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
            for example in encoded[
                update * args.accumulation : (update + 1) * args.accumulation
            ]:
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
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
