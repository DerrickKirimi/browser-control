"""Sample several screenshot rollouts per task and record what varies. No training."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from browser_control.lfm3_vision import (
    MODEL_ID,
    MODEL_REVISION,
    LFM3VisionPolicy,
    adapter_hash,
    attach_adapter,
)
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.transformers_policy import DecoderConfig
from browser_control.vision_browser import VisionBrowserGymEnvironment
from browser_control.vision_evaluation import (
    VISION_TASKS,
    classify_variation_group,
    collect_vision_episode,
)


def generation_seed(task_seed: int, member: int) -> int:
    return 1_000_000 + task_seed * 64 + member * 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=VISION_TASKS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier evidence")
    if (
        len(set(args.seeds)) != len(args.seeds)
        or any(not 100 <= seed < 10000 for seed in args.seeds)
        or not 2 <= args.group_size <= 8
        or not 0 < args.top_p <= 1
    ):
        parser.error("Use distinct training seeds 100..9999, groups of 2..8, top_p (0, 1]")
    decoder = DecoderConfig(
        "variation-sampling", True, 128, temperature=1.0, top_p=args.top_p
    )
    snapshot = Path("source-manifest.json")
    source = json.loads(snapshot.read_text()) if snapshot.exists() else None
    if source is not None:
        for name, digest in source["files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError("Source hash differs: " + name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "running",
        "scope": "within-group native variation of one visual task; no training",
        "source_snapshot": source,
        "manifest": RunManifest(
            "lfm3-vision-variation",
            23,
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "task": args.task,
                "adapter_sha256": adapter_hash(args.adapter),
                "task_seeds": args.seeds,
                "group_size": args.group_size,
                "decoder": asdict(decoder),
                "image_condition": "screenshot",
                "max_steps": 1,
                "max_prompt_tokens": 2048,
                "enable_thinking": False,
                "attention": "sdpa",
                "dtype": "float16",
                "generation_seed": "1000000 + task_seed * 64 + member * 16",
            },
        ).to_dict(),
        "groups": [],
        "runtime_versions": {
            name: version(name)
            for name in ("torch", "torchvision", "transformers", "tokenizers")
        },
    }

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        policy = LFM3VisionPolicy.load(decoder)
        attach_adapter(policy, args.adapter)
        payload["effective_generation_config"] = (
            policy.model.generation_config.to_dict()
        )
        policy.torch.cuda.reset_peak_memory_stats()
        root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
        with MiniwobAssetServer(root) as server:
            with closing(
                VisionBrowserGymEnvironment.miniwob(
                    args.task, miniwob_url=server.base_url
                )
            ) as env:
                for seed in args.seeds:
                    rows = [
                        collect_vision_episode(
                            env,
                            policy,
                            seed,
                            "screenshot",
                            args.output.parent / "screenshots",
                            generation_seed=generation_seed(seed, member),
                            record_logprobs=True,
                        )
                        for member in range(args.group_size)
                    ]
                    payload["groups"].append(
                        {"episodes": rows, "summary": classify_variation_group(rows)}
                    )
                    save()
        payload.update(
            status="complete",
            peak_allocated_bytes=policy.torch.cuda.max_memory_allocated(),
        )
    except BaseException as error:
        payload.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
