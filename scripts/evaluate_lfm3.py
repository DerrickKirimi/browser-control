"""Run the frozen LFM3 port gate on local click-test; no optimizer is created."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from browser_control.actions import parse_click_test_action
from browser_control.evaluation import evaluate_click_test, wilson_interval
from browser_control.lfm3 import (
    MODEL_ID,
    MODEL_REVISION,
    LFM3Policy,
    build_lfm3_messages,
    frozen_gate,
    parse_lfm3_action,
)
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.prompts import build_click_test_messages
from browser_control.transformers_policy import DecoderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--assets-dir", type=Path, default=Path("artifacts/miniwob-plusplus")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/lfm3/click-test/frozen.json")
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve it and choose a new path")
    if args.episodes < 1:
        parser.error("episodes must be positive")

    policy = LFM3Policy.load(DecoderConfig("greedy", False, 128), "strict-text")
    source_manifest = Path("source-manifest.json")
    payload = {
        "status": "running",
        "source_snapshot": json.loads(source_manifest.read_text())
        if source_manifest.exists()
        else None,
        "generation_config": policy.model.generation_config.to_dict(),
        "manifest": RunManifest(
            run_name="lfm3-click-test-frozen-port",
            seed=23,
            config={
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "scope": "one seed-invariant task; sampling uncertainty only",
                "training": False,
                "episodes_per_sampling_cell": args.episodes,
            },
        ).to_dict(),
        "conditions": [],
    }
    root = ensure_miniwob_assets(args.assets_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with MiniwobAssetServer(root) as server:

        def factory():
            return LocalBrowserGymEnvironment.miniwob(
                "click-test", miniwob_url=server.base_url
            )

        for interface in ("strict-text", "native-tools"):
            policy.interface = interface
            builder = (
                build_lfm3_messages
                if interface == "native-tools"
                else build_click_test_messages
            )
            action_parser = (
                parse_lfm3_action
                if interface == "native-tools"
                else parse_click_test_action
            )
            env = factory()
            try:
                messages = builder(env.reset(seed=23).observation)
                prompt_hash = hashlib.sha256(
                    json.dumps(messages, sort_keys=True).encode()
                ).hexdigest()
                encoded_ids = policy.encode(messages)["input_ids"].tolist()
                encoded_prompt_hash = hashlib.sha256(
                    json.dumps(encoded_ids).encode()
                ).hexdigest()
                numerical = policy.check_logits(messages)
            finally:
                env.close()
            for mode in ("greedy", "sampling"):
                # XML needs a larger budget than the historical 32-token action.
                policy.decoder = DecoderConfig(
                    mode,
                    mode == "sampling",
                    128,
                    temperature=1.0 if mode == "sampling" else None,
                    top_p=0.95 if mode == "sampling" else None,
                )
                for seed in (23, 47, 71):
                    policy.torch.cuda.reset_peak_memory_stats()
                    report = evaluate_click_test(
                        factory,
                        policy,
                        range(
                            seed, seed + (args.episodes if mode == "sampling" else 1)
                        ),
                        message_builder=builder,
                        action_parser=action_parser,
                    )
                    successes = sum(episode.success for episode in report.episodes)
                    cell = {
                        "interface": interface,
                        "decoder": asdict(policy.decoder),
                        "seed": seed,
                        "prompt_sha256": prompt_hash,
                        "encoded_prompt_sha256": encoded_prompt_hash,
                        "numerical_check": numerical,
                        "evaluation": report.to_dict(),
                        "peak_vram_bytes": policy.torch.cuda.max_memory_allocated(),
                        "wilson_95": wilson_interval(successes, len(report.episodes))
                        if mode == "sampling"
                        else None,
                    }
                    payload["conditions"].append(cell)
                    temporary = args.output.with_suffix(".tmp")
                    temporary.write_text(json.dumps(payload, indent=2) + "\n")
                    temporary.replace(args.output)
                    print(interface, mode, seed, report.success_rate, flush=True)
    payload["status"] = "complete"
    payload["gate"] = frozen_gate(payload["conditions"])
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("Frozen gate:", payload["gate"])
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
