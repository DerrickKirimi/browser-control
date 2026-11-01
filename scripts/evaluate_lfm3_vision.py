"""Bounded frozen LFM3-VL one-click development evaluation on a T4."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from browser_control.lfm3_vision import MODEL_ID, MODEL_REVISION, LFM3VisionPolicy
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.transformers_policy import DecoderConfig
from browser_control.vision_browser import VisionBrowserGymEnvironment
from browser_control.vision_evaluation import (
    IMAGE_CONDITIONS,
    VISION_TASKS,
    collect_vision_episode,
    summarize_vision_episodes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks", choices=VISION_TASKS, nargs="+", default=list(VISION_TASKS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30, 60)))
    parser.add_argument("--decoder", choices=("greedy", "sampling"), required=True)
    parser.add_argument("--image-condition", choices=IMAGE_CONDITIONS, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier evaluation evidence")
    if (
        len(args.seeds) > 30
        or len(set(args.seeds)) != len(args.seeds)
        or any(not 0 <= seed < 10000 for seed in args.seeds)
        or len(set(args.tasks)) != len(args.tasks)
    ):
        parser.error(
            "Use distinct development tasks and at most30distinct seeds0..9999"
        )
    decoder = (
        DecoderConfig("greedy", False, 128)
        if args.decoder == "greedy"
        else DecoderConfig("sampling", True, 128, temperature=1.0, top_p=0.95)
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
        "scope": "frozen one-click vision development; no GRPO result",
        "source_snapshot": source,
        "manifest": RunManifest(
            "lfm3-vision-frozen",
            23,
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tasks": args.tasks,
                "task_seeds": args.seeds,
                "decoder": asdict(decoder),
                "image_condition": args.image_condition,
                "max_steps": 1,
                "max_prompt_tokens": 2048,
                "enable_thinking": False,
                "attention": "sdpa",
                "dtype": "float16",
                "generation_seed": "task_seed * 16",
                "screenshot_coordinates": "normalized0..1000 mapped to inclusive screenshot pixels; BrowserGym handles page scaling",
            },
        ).to_dict(),
        "conditions": [],
        "runtime_versions": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "tokenizers",
                "safetensors",
                "numpy",
                "pillow",
                "browsergym-core",
                "browsergym-miniwob",
                "playwright",
                "gymnasium",
                "greenlet",
            )
        },
    }

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        policy = LFM3VisionPolicy.load(decoder)
        payload["effective_generation_config"] = (
            policy.model.generation_config.to_dict()
        )
        policy.torch.cuda.reset_peak_memory_stats()
        root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
        with MiniwobAssetServer(root) as server:
            for task in args.tasks:
                directory = Path("screenshots") / task
                condition = {
                    "task": task,
                    "screenshot_directory": directory.as_posix(),
                    "episodes": [],
                }
                payload["conditions"].append(condition)
                save()
                with closing(
                    VisionBrowserGymEnvironment.miniwob(
                        task, miniwob_url=server.base_url
                    )
                ) as env:
                    for seed in args.seeds:
                        condition["episodes"].append(
                            collect_vision_episode(
                                env,
                                policy,
                                seed,
                                args.image_condition,
                                args.output.parent / directory,
                            )
                        )
                        save()
                condition["summary"] = summarize_vision_episodes(condition["episodes"])
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
