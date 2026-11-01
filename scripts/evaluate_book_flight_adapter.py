"""Native book-flight evaluation of one saved adapter on the comparison cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from browser_control.book_flight import BOOK_FLIGHT_TOOLS
from browser_control.book_flight_curriculum import validate_initial_adapter
from browser_control.book_flight_diagnostic import (
    COMPARISON_SEEDS,
    SAMPLING_128,
    summarize_rollouts,
)
from browser_control.book_flight_evaluation import collect_book_flight_episode
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION, LFM3Policy
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the previous evaluation")
    from peft import PeftModel
    from preflight_lfm3_attention import LORA

    validate_initial_adapter(
        json.loads((args.adapter / "adapter_config.json").read_text()),
        LORA,
        revision=MODEL_REVISION,
        model_id=MODEL_ID,
    )
    snapshot_path = Path("source-manifest.json")
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    if snapshot is not None:
        for name, expected in snapshot["files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "running",
        "scope": "matched development comparison; not a held-out estimate",
        "source_snapshot": snapshot,
        "manifest": RunManifest(
            "lfm3-book-flight-adapter-evaluation",
            23,
            {
                "adapter_sha256": hashlib.sha256(
                    (args.adapter / "adapter_model.safetensors").read_bytes()
                ).hexdigest(),
                "model_revision": MODEL_REVISION,
                "task_seeds": list(COMPARISON_SEEDS),
                "decoder_name": "SAMPLING_128",
                "decoder": asdict(SAMPLING_128),
                "attention": "sdpa",
                "max_prompt_tokens": 16384,
                "prefill_chunk_size": 512,
                "max_steps": 15,
                "enable_thinking": False,
                "generation_seed": "task_seed * 16 + turn",
            },
        ).to_dict(),
        "episodes": [],
    }

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        policy = LFM3Policy.load(SAMPLING_128, "native-tools", "sdpa")
        policy.tools = BOOK_FLIGHT_TOOLS
        policy.max_prompt_tokens = 16384
        policy.model = PeftModel.from_pretrained(policy.model, args.adapter).eval()
        policy.model.generation_config.prefill_chunk_size = 512
        if any(p.requires_grad for p in policy.model.parameters()):
            raise ValueError("Evaluation must keep all model parameters frozen")
        root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
        with (
            MiniwobAssetServer(root) as server,
            closing(
                LocalBrowserGymEnvironment.miniwob(
                    "book-flight-nodelay", miniwob_url=server.base_url
                )
            ) as env,
        ):
            for seed in COMPARISON_SEEDS:
                payload["episodes"].append(
                    collect_book_flight_episode(
                        env, policy, seed, numerical_check=policy.check_logits
                    )
                )
                save()
        payload["summary"] = summarize_rollouts(payload["episodes"])
        payload["status"] = "complete"
    except BaseException as error:
        payload.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
