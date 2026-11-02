"""Development-only mode diagnostic: teacher-forced probes and rollouts per cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from browser_control.book_flight import BOOK_FLIGHT_TOOLS
from browser_control.book_flight_diagnostic import (
    CELLS,
    DEVELOPMENT_SEEDS,
    FIRST_TURN_VALID_RANGE,
    PARSERS,
    decide,
    score_probe,
    summarize_probes,
    summarize_rollouts,
)
from browser_control.book_flight_evaluation import collect_book_flight_episode
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION, LFM3Policy
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import (
    MINIWOB_REVISION,
    MiniwobAssetServer,
    ensure_miniwob_assets,
)

MAX_PROMPT_TOKENS = 16384
PREFILL_CHUNK_SIZE = 512
PROBE_SEED_OFFSET = 100000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--attention", choices=("sdpa", "eager"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve it before starting a separate run")
    probe_report = json.loads(args.probes.read_text())
    probes = probe_report["probes"]
    if {probe["task_seed"] for probe in probes} - set(DEVELOPMENT_SEEDS):
        parser.error("probes must come from development seeds only")
    seeds = list(DEVELOPMENT_SEEDS)
    config = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "miniwob_revision": MINIWOB_REVISION,
        "task": "book-flight-nodelay",
        "task_seeds": seeds,
        "probes_sha256": hashlib.sha256(args.probes.read_bytes()).hexdigest(),
        "generation_seed": "task_seed * 16 + turn (zero-based)",
        "probe_seed": f"{PROBE_SEED_OFFSET} + task_seed * 16 + probe index",
        "max_steps": 15,
        "attention": args.attention,
        "prefill_chunk_size": PREFILL_CHUNK_SIZE,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "training": False,
        "tools": BOOK_FLIGHT_TOOLS,
        "cells": {
            name: {**cell, "decoder": asdict(cell["decoder"])}
            for name, cell in CELLS.items()
        },
    }
    source = Path("source-manifest.json")
    payload = {
        "status": "running",
        "source_snapshot": json.loads(source.read_text()) if source.exists() else None,
        "manifest": RunManifest(
            "lfm3-book-flight-diagnostic", seeds[0], config
        ).to_dict(),
        "cells": {name: {"probes": [], "episodes": []} for name in CELLS},
    }
    if args.resume_from is not None:
        previous = json.loads(args.resume_from.read_text())
        if previous["manifest"]["config"] != config:
            parser.error("resume requires an identical diagnostic configuration")
        retained_source = previous["source_snapshot"]["files_sha256"]
        current_source = payload["source_snapshot"]["files_sha256"]
        if {k: v for k, v in retained_source.items() if k.startswith("src/")} != {
            k: v for k, v in current_source.items() if k.startswith("src/")
        }:
            parser.error("resume requires unchanged src/ hashes")
        payload["cells"] = previous["cells"]
        payload["resumed_from"] = {
            "report_sha256": hashlib.sha256(args.resume_from.read_bytes()).hexdigest()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output)

    first = next(iter(CELLS.values()))
    policy = LFM3Policy.load(first["decoder"], "native-tools", args.attention)
    policy.tools = BOOK_FLIGHT_TOOLS
    policy.model.generation_config.prefill_chunk_size = PREFILL_CHUNK_SIZE
    policy.max_prompt_tokens = MAX_PROMPT_TOKENS
    payload["generation_config"] = policy.model.generation_config.to_dict()
    save()
    root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
    with MiniwobAssetServer(root) as server:
        env = LocalBrowserGymEnvironment.miniwob(
            "book-flight-nodelay", miniwob_url=server.base_url
        )
        try:
            for name, cell in CELLS.items():
                policy.decoder = cell["decoder"]
                policy.enable_thinking = cell["enable_thinking"]
                results = payload["cells"][name]
                policy.torch.cuda.reset_peak_memory_stats()
                for index, probe in enumerate(probes):
                    if index < len(results["probes"]):
                        continue
                    messages = [
                        {"role": "system", "content": cell["system_prompt"]},
                        *probe["messages"][1:],
                    ]
                    seed = PROBE_SEED_OFFSET + probe["task_seed"] * 16 + index
                    output = policy.generate(messages, seed=seed)
                    results["probes"].append(
                        {
                            "task_seed": probe["task_seed"],
                            "skill": probe["skill"],
                            "prior_error": probe["prior_error"],
                            "generation_seed": seed,
                            **asdict(output),
                            **score_probe(probe, output.text, cell["parser"]),
                        }
                    )
                    save()
                    print(name, "probe", index, probe["skill"], flush=True)
                for seed in seeds[len(results["episodes"]) :]:
                    episode = collect_book_flight_episode(
                        env,
                        policy,
                        seed,
                        numerical_check=policy.check_logits,
                        parse_output=PARSERS[cell["parser"]],
                        system_prompt=cell["system_prompt"],
                    )
                    results["episodes"].append(episode)
                    save()
                    print(
                        name,
                        seed,
                        episode["success"],
                        episode["stop_reason"],
                        episode["environment_steps"],
                        flush=True,
                    )
                results["peak_vram_bytes"] = policy.torch.cuda.max_memory_allocated()
                results["summary"] = {
                    "probes": summarize_probes(results["probes"]),
                    "rollouts": summarize_rollouts(results["episodes"]),
                }
                payload["decision"] = decide(
                    {
                        key: value["summary"]
                        for key, value in payload["cells"].items()
                        if "summary" in value
                    }
                )
                save()
                print(name, json.dumps(results["summary"]["rollouts"]), flush=True)
                low, high = FIRST_TURN_VALID_RANGE
                first_turn = results["summary"]["rollouts"]["first_turn_valid"]
                if name == "C0" and not low <= first_turn <= high:
                    payload["decision"] = decide(
                        {"C0": results["summary"], "C2": results["summary"]}
                    )
                    break  # The rule draws no conclusions; spend no more GPU time.
        finally:
            env.close()
    payload["status"] = "complete"
    save()
    print(json.dumps(payload["decision"]), flush=True)


if __name__ == "__main__":
    main()
