"""Compare a saved adapter on matched original and consistently relabeled IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from browser_control.book_flight import BOOK_FLIGHT_TOOLS
from browser_control.book_flight_diagnostic import (
    CONFIRMATION_SEEDS,
    DECISION_SKILLS,
    DEVELOPMENT_SEEDS,
    SAMPLING_128,
    build_probes,
    score_probe,
    summarize_probes,
    summarize_rollouts,
)
from browser_control.book_flight_evaluation import collect_book_flight_episode
from browser_control.book_flight_ids import compare_id_scores, validate_probe_pair
from browser_control.book_flight_sft import TRAINING_SEEDS, expert_examples
from browser_control.lfm3 import MODEL_REVISION, LFM3Policy
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets


def main():
    from peft import PeftModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--permuted-probes", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-corpus", type=Path)
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the previous evaluation")
    source = json.loads(args.probes.read_text())
    altered = json.loads(args.permuted_probes.read_text())
    validate_probe_pair(
        source,
        altered,
        original_sha256=hashlib.sha256(args.probes.read_bytes()).hexdigest(),
    )
    probes = source["probes"]
    if len(probes) != len(altered["probes"]) or len(probes) != len(altered["mappings"]):
        parser.error("Paired probe lengths differ")
    cohort = CONFIRMATION_SEEDS if args.confirmation else DEVELOPMENT_SEEDS
    for original in probes:
        if original["task_seed"] not in cohort:
            parser.error("Only fixed development probes are allowed")
    if args.confirmation:
        pairs = [
            (p["task_seed"], p["skill"])
            for p in probes
            if p["skill"] in DECISION_SKILLS
        ]
        if (
            source.get("status") != "complete"
            or len(pairs) != 120
            or set(pairs)
            != {(seed, skill) for seed in cohort for skill in DECISION_SKILLS}
            or args.training_corpus is not None
        ):
            parser.error(
                "Confirmation requires complete seeds50–69, six skills each, no training corpus"
            )
    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
    if adapter_config.get("revision") not in (None, MODEL_REVISION):
        parser.error("Adapter config names a different base revision")
    snapshot_path = Path("source-manifest.json")
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    if snapshot is not None:
        for name, expected in snapshot["files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")
    config = {
        "model_revision": MODEL_REVISION,
        "decoder": asdict(SAMPLING_128),
        "attention": "sdpa",
        "prefill_chunk_size": 512,
        "max_prompt_tokens": 16384,
        "training": False,
        "confirmation": args.confirmation,
        "probe_seed": "100000 + task_seed * 16 + probe_index",
        "permutation_seed": altered["permutation_seed"],
        "scheme": altered["scheme"],
        "inputs_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("probes", args.probes),
                ("permuted_probes", args.permuted_probes),
                ("adapter_config", args.adapter / "adapter_config.json"),
                ("adapter_weights", args.adapter / "adapter_model.safetensors"),
            )
        },
    }
    stages = [("original", probes), ("permuted", altered["probes"])]
    if args.training_corpus is not None:
        corpus = json.loads(args.training_corpus.read_text())
        if corpus["status"] != "complete":
            parser.error("Seen-state comparison requires a complete training corpus")
        reconstructed = [
            e for record in corpus["records"] for e in expert_examples(record)
        ]
        if reconstructed != corpus["examples"]:
            parser.error(
                "Training corpus does not match clean successful expert records"
            )
        training_probes = [
            probe
            for record in corpus["records"]
            for probe in build_probes(record)
            if probe["skill"] in ("choose_date", "book_flight")
        ]
        if any(probe["task_seed"] not in TRAINING_SEEDS for probe in training_probes):
            parser.error("Seen-state comparison must use training seeds only")
        stages.append(("seen_training", training_probes))
        config["seen_training_corpus_sha256"] = hashlib.sha256(
            args.training_corpus.read_bytes()
        ).hexdigest()
        config["seen_training_scope"] = (
            "date/flight probes from training states; not generalization"
        )
    payload = {
        "status": "running",
        "source_snapshot": snapshot,
        "manifest": RunManifest("lfm3-book-flight-id-probes", 23, config).to_dict(),
        "original": [],
        "permuted": [],
    }
    if args.training_corpus is not None:
        payload["seen_training"] = []
    if args.confirmation:
        payload["native_confirmation"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    # Load the pinned base explicitly even when an old adapter config has no revision.
    policy = LFM3Policy.load(SAMPLING_128, "native-tools", "sdpa")
    policy.tools = BOOK_FLIGHT_TOOLS
    policy.max_prompt_tokens = 16384
    policy.model = PeftModel.from_pretrained(policy.model, args.adapter).eval()
    policy.model.generation_config.prefill_chunk_size = 512
    if any(p.requires_grad for p in policy.model.parameters()):
        raise ValueError("Probe evaluation must keep all model parameters frozen")
    try:
        for stage, rows in stages:
            for index, probe in enumerate(rows):
                seed = 100000 + probe["task_seed"] * 16 + index
                output = policy.generate(probe["messages"], seed=seed)
                payload[stage].append(
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
            payload[stage + "_summary"] = summarize_probes(payload[stage])
            if stage == "permuted":
                payload["decision"] = compare_id_scores(
                    payload["original_summary"],
                    payload["permuted_summary"],
                    skills=DECISION_SKILLS if args.confirmation else None,
                )
            save()
        payload["decision"] = compare_id_scores(
            payload["original_summary"],
            payload["permuted_summary"],
            skills=DECISION_SKILLS if args.confirmation else None,
        )
        if args.confirmation:
            root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
            with MiniwobAssetServer(root) as server:
                env = LocalBrowserGymEnvironment.miniwob(
                    "book-flight-nodelay", miniwob_url=server.base_url
                )
                try:
                    for seed in list(CONFIRMATION_SEEDS)[:10]:
                        payload["native_confirmation"].append(
                            collect_book_flight_episode(
                                env, policy, seed, numerical_check=policy.check_logits
                            )
                        )
                        save()
                finally:
                    env.close()
            payload["native_confirmation_summary"] = summarize_rollouts(
                payload["native_confirmation"]
            )
            skills_pass = all(
                payload["original_summary"]["skills"][skill]["correct"] >= 14
                for skill in DECISION_SKILLS
            )
            native_pass = (
                sum(row["success"] for row in payload["native_confirmation"]) >= 2
            )
            payload["confirmation_qualification"] = {
                "passed": skills_pass
                and native_pass
                and not payload["decision"]["stop_grpo"],
                "skills_pass": skills_pass,
                "native_pass": native_pass,
                "id_pass": not payload["decision"]["stop_grpo"],
                "scope": "fresh development qualification; not held-out generalization",
            }
        payload["status"] = "complete"
    except BaseException as error:
        payload.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
        raise
    finally:
        save()
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
