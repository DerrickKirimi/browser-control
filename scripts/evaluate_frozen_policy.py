"""Evaluate the untouched LFM2 policy on seeded local click-test episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from browser_control.evaluation import evaluate_click_test
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.transformers_policy import DEFAULT_MODEL_ID, TransformersPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=23)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("artifacts/miniwob-plusplus"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/frozen-policy.json"),
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    document_root = ensure_miniwob_assets(args.assets_dir)
    policy = TransformersPolicy.load(args.model_id)
    policy.torch.cuda.reset_peak_memory_stats(policy.device.index)
    seeds = range(args.first_seed, args.first_seed + args.episodes)

    with MiniwobAssetServer(document_root) as server:
        report = evaluate_click_test(
            lambda: LocalBrowserGymEnvironment.miniwob(
                "click-test", miniwob_url=server.base_url
            ),
            policy,
            seeds,
        )

    payload = {
        "model_id": args.model_id,
        "device": policy.device.to_dict(),
        "peak_vram_bytes": policy.torch.cuda.max_memory_allocated(
            policy.device.index
        ),
        "evaluation": report.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
