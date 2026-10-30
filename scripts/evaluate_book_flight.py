"""Frozen native-tool book-flight evaluation; pilot and held-out seeds are disjoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from browser_control.book_flight import BOOK_FLIGHT_TOOLS, SYSTEM_PROMPT
from browser_control.book_flight_evaluation import collect_book_flight_episode
from browser_control.evaluation import wilson_interval
from browser_control.evaluation_resume import validate_frozen_resume
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION, LFM3Policy
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import (
    MINIWOB_REVISION,
    MiniwobAssetServer,
    ensure_miniwob_assets,
)
from browser_control.transformers_policy import DecoderConfig

# Measure scripted development traces before freezing this budget. No truncation.
MAX_PROMPT_TOKENS = 16384
DECODER = DecoderConfig("sampling", True, 128, temperature=1.0, top_p=0.95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("pilot", "heldout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve it before starting a separate run")
    seeds = range(20, 23) if args.split == "pilot" else range(10000, 10100)
    policy = LFM3Policy.load(DECODER, "native-tools")
    policy.tools = BOOK_FLIGHT_TOOLS
    policy.model.generation_config.prefill_chunk_size = 512
    policy.max_prompt_tokens = MAX_PROMPT_TOKENS
    source = Path("source-manifest.json")
    payload = {
        "status": "running",
        "split": args.split,
        "source_snapshot": json.loads(source.read_text()) if source.exists() else None,
        "manifest": RunManifest(
            "lfm3-book-flight-frozen-" + args.split,
            seeds.start,
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "miniwob_revision": MINIWOB_REVISION,
                "task": "book-flight-nodelay",
                "task_seeds": list(seeds),
                "generation_seed": "task_seed * 16 + turn (zero-based)",
                "max_steps": 15,
                "prefill_chunk_size": 512,
                "max_prompt_tokens": MAX_PROMPT_TOKENS,
                "decoder": asdict(DECODER),
                "training": False,
                "tools": BOOK_FLIGHT_TOOLS,
                "system_prompt": SYSTEM_PROMPT,
                "invalid_action": "terminate without stepping; reward zero",
                "action_error": "continue with error and observed state within the same horizon",
                "reward": "native MiniWoB reward; success = terminal and reward > 0",
            },
        ).to_dict(),
        "generation_config": policy.model.generation_config.to_dict(),
        "episodes": [],
    }
    retained = 0
    if args.resume_from is not None:
        previous = json.loads(args.resume_from.read_text())
        retained = validate_frozen_resume(previous, payload)
        payload["episodes"] = previous["episodes"]
        payload["resumed_from"] = {
            "report_sha256": hashlib.sha256(args.resume_from.read_bytes()).hexdigest(),
            "retained_episodes": retained,
            "manifest": previous["manifest"],
            "source_snapshot": previous["source_snapshot"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
    with MiniwobAssetServer(root) as server:
        env = LocalBrowserGymEnvironment.miniwob(
            "book-flight-nodelay", miniwob_url=server.base_url
        )
        try:
            for seed in list(seeds)[retained:]:
                policy.torch.cuda.reset_peak_memory_stats()
                episode = collect_book_flight_episode(
                    env, policy, seed, numerical_check=policy.check_logits
                )
                episode["peak_vram_bytes"] = policy.torch.cuda.max_memory_allocated()
                payload["episodes"].append(episode)
                save()
                print(
                    seed,
                    episode["success"],
                    episode["stop_reason"],
                    episode["environment_steps"],
                    flush=True,
                )
        finally:
            env.close()
    count = len(payload["episodes"])
    successes = sum(episode["success"] for episode in payload["episodes"])
    payload.update(
        status="complete",
        success_rate=successes / count,
        wilson_95=wilson_interval(successes, count),
    )
    if args.split == "heldout":
        assert [episode["task_seed"] for episode in payload["episodes"]] == list(
            range(10000, 10100)
        )
        payload["gate"] = {
            "eligible": True,
            "direct_grpo": successes >= 10,
            "decision": "direct-grpo"
            if successes >= 10
            else "sft-warm-start-before-grpo",
        }
    save()
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key in ("status", "success_rate", "wilson_95", "gate")
            }
        )
    )


if __name__ == "__main__":
    main()
