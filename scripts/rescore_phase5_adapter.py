"""Paired frozen/trained evaluation of the existing Phase 5 adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from browser_control.evaluation import evaluate_click_test
from browser_control.hardware import inspect_cuda_device
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.rescoring import GATE_A_DECODERS, summarize_rescore
from browser_control.transformers_policy import DEFAULT_MODEL_ID, TransformersPolicy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_conditions(
    policy: TransformersPolicy,
    environment_factory: Any,
    sample_seeds: tuple[int, ...],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for decoder in GATE_A_DECODERS:
        policy.decoder = decoder
        seeds = sample_seeds if decoder.do_sample else (sample_seeds[0],)
        report = evaluate_click_test(environment_factory, policy, seeds)
        results[decoder.name] = summarize_rescore(decoder, report).to_dict()
    return results


def run(
    *,
    adapter_dir: Path,
    output: Path,
    assets_dir: Path,
    model_id: str,
    samples: int,
    first_seed: int,
) -> dict[str, object]:
    if samples < 50:
        raise ValueError("Gate A requires at least 50 stochastic samples")
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"adapter weights not found: {weights_path}")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the environment, model, and training extras") from error

    device = inspect_cuda_device(torch)
    dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(f"cuda:{device.index}")
    model.eval()
    document_root = ensure_miniwob_assets(assets_dir)
    sample_seeds = tuple(range(first_seed, first_seed + samples))
    manifest = RunManifest(
        run_name="phase5-gate-a-rescore",
        seed=first_seed,
        config={
            "adapter_dir": str(adapter_dir),
            "adapter_sha256": _sha256(weights_path),
            "model_id": model_id,
            "samples_per_stochastic_condition": samples,
            "sample_seeds": list(sample_seeds),
            "task": "miniwob.click-test",
            "task_scope": "one seed-invariant prompt; not held-out generalization",
        },
    )

    with MiniwobAssetServer(document_root) as server:
        def environment_factory() -> LocalBrowserGymEnvironment:
            return LocalBrowserGymEnvironment.miniwob(
                "click-test", miniwob_url=server.base_url
            )

        frozen_policy = TransformersPolicy(model, tokenizer, torch, device)
        frozen = _evaluate_conditions(
            frozen_policy, environment_factory, sample_seeds
        )
        trained_model = PeftModel.from_pretrained(model, adapter_dir)
        trained_model.to(f"cuda:{device.index}")
        trained_model.eval()
        trained_policy = TransformersPolicy(trained_model, tokenizer, torch, device)
        trained = _evaluate_conditions(
            trained_policy, environment_factory, sample_seeds
        )

    payload: dict[str, object] = {
        "manifest": manifest.to_dict(),
        "device": device.to_dict(),
        "dtype": "float32",
        "frozen": frozen,
        "trained": trained,
        "success_rate_deltas": {
            name: trained[name]["success_rate"] - frozen[name]["success_rate"]
            for name in frozen
        },
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device.index)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=2000)
    parser.add_argument(
        "--assets-dir", type=Path, default=Path("artifacts/miniwob-plusplus")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase5/gate-a/phase5-gate-a-rescore.json"),
    )
    args = parser.parse_args()
    report = run(
        adapter_dir=args.adapter_dir,
        output=args.output,
        assets_dir=args.assets_dir,
        model_id=args.model_id,
        samples=args.samples,
        first_seed=args.first_seed,
    )
    compact = {
        "output": str(args.output),
        "conditions": {
            name: {
                "frozen": f"{report['frozen'][name]['successes']}/{report['frozen'][name]['samples']}",
                "trained": f"{report['trained'][name]['successes']}/{report['trained'][name]['samples']}",
                "success_rate_delta": report["success_rate_deltas"][name],
            }
            for name in report["frozen"]
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
