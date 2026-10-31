"""Collect frozen LFM3/browser token rollouts before attempting real GRPO.

Training seeds only. This checks raw history identity, aligned masks and native
reward variation; it performs no optimizer update and cannot validate TRL memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from browser_control.book_flight import (
    initial_messages,
    observation_text,
    parse_book_flight_action,
)
from browser_control.book_flight_curriculum import validate_initial_adapter
from browser_control.book_flight_rollout import (
    TransformersTokenSampler,
    chat_ids,
    collect_token_episode,
)
from browser_control.book_flight_sft import TRAINING_SEEDS, encode_example
from browser_control.environment import BrowserObservation
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION, LFM3Policy
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.manifest import RunManifest
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.transformers_policy import DecoderConfig


def audit_history(episode, tokenizer):
    """Compare actual sampled contexts/IDs with canonical evaluation histories."""
    combined = episode["prompt_ids"] + episode["completion_ids"]
    messages = initial_messages(
        BrowserObservation(**episode["steps"][0]["observation"])
    )
    rows = []
    for index, step in enumerate(episode["steps"]):
        if index:
            messages.append(
                {
                    "role": "tool",
                    "content": observation_text(
                        BrowserObservation(**step["observation"])
                    ),
                }
            )
        expected = chat_ids(tokenizer, messages, generation=True)
        actual = combined[: step["prompt_tokens"]]
        row = {"turn": step["turn"], "context_ids_match": actual == expected}
        if step["valid_action"]:
            action = parse_book_flight_action(step["text"])
            encoded = encode_example(
                tokenizer,
                {"messages": messages, "assistant": action.message()},
                max_tokens=max(16384, len(combined) * 2 + 1024),
            )
            row["assistant_ids_match"] = (
                step["token_ids"] == encoded["input_ids"][encoded["prompt_tokens"] :]
            )
            messages.append(action.message())
        rows.append(row)
    return rows


def main():
    from peft import PeftModel
    from preflight_lfm3_attention import LORA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 132])
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, choices=(1.0, 1.5), default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier token-rollout evidence")
    if not 2 <= args.group_size <= 4 or not 1 <= len(args.seeds) <= 4:
        parser.error("Bounded diagnostic needs 1..4 seeds and group size 2..4")
    if len(set(args.seeds)) != len(args.seeds) or any(
        seed not in TRAINING_SEEDS for seed in args.seeds
    ):
        parser.error("Use distinct training seeds 100..9999 only")
    validate_initial_adapter(
        json.loads((args.adapter / "adapter_config.json").read_text()),
        LORA,
        revision=MODEL_REVISION,
        model_id=MODEL_ID,
    )
    decoder = DecoderConfig(
        "token-rollout", True, 128, temperature=args.temperature, top_p=0.95
    )
    snapshot_path = Path("source-manifest.json")
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    if snapshot is not None:
        for name, expected in snapshot["files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")
    payload = {
        "status": "running",
        "source_snapshot": snapshot,
        "scope": "frozen raw-token collection; no TRL optimizer or task-learning claim",
        "manifest": RunManifest(
            "lfm3-book-flight-token-rollouts",
            23,
            {
                "model_revision": MODEL_REVISION,
                "decoder": asdict(decoder),
                "task_seeds": args.seeds,
                "group_size": args.group_size,
                "generation_seed": "1000000 + seed * 64 + member * 16 + turn",
                "adapter_sha256": hashlib.sha256(
                    (args.adapter / "adapter_model.safetensors").read_bytes()
                ).hexdigest(),
            },
        ).to_dict(),
        "episodes": [],
        "groups": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        policy = LFM3Policy.load(decoder, "native-tools", "sdpa")
        policy.model = PeftModel.from_pretrained(policy.model, args.adapter).eval()
        policy.model.generation_config.prefill_chunk_size = 512
        if any(parameter.requires_grad for parameter in policy.model.parameters()):
            raise ValueError("Frozen collection must not enable gradients")
        sampler = TransformersTokenSampler(
            policy.model, policy.tokenizer, policy.torch, decoder
        )
        policy.torch.cuda.reset_peak_memory_stats()
        root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
        with MiniwobAssetServer(root) as server:
            env = LocalBrowserGymEnvironment.miniwob(
                "book-flight-nodelay", miniwob_url=server.base_url
            )
            try:
                for seed in args.seeds:
                    group = []
                    for member in range(args.group_size):
                        row = collect_token_episode(
                            env,
                            sampler,
                            policy.tokenizer,
                            seed,
                            generation_seed=1_000_000 + seed * 64 + member * 16,
                        )
                        row["history_audit"] = audit_history(row, policy.tokenizer)
                        if group and row["prompt_ids"] != group[0]["prompt_ids"]:
                            raise ValueError(
                                "Grouped reset changed the initial prompt IDs"
                            )
                        if not (
                            len(row["completion_ids"])
                            == len(row["env_mask"])
                            == len(row["logprobs"])
                        ):
                            raise ValueError(
                                "Rollout IDs, masks and logprobs are misaligned"
                            )
                        if any(
                            logprob != 0.0
                            for logprob, mask in zip(row["logprobs"], row["env_mask"])
                            if not mask
                        ):
                            raise ValueError("External tokens must carry zero logprobs")
                        group.append(row)
                        payload["episodes"].append(row)
                        save()
                    rewards = [float(row["native_success"]) for row in group]
                    payload["groups"].append(
                        {
                            "task_seed": seed,
                            "native_rewards": rewards,
                            "reward_variation": len(set(rewards)) > 1,
                        }
                    )
                    save()
            finally:
                env.close()
        audits = [row for e in payload["episodes"] for row in e["history_audit"]]
        payload.update(
            status="complete",
            canonical_history_ids_match=all(
                row["context_ids_match"] and row.get("assistant_ids_match", True)
                for row in audits
            ),
            groups_with_native_variation=sum(
                group["reward_variation"] for group in payload["groups"]
            ),
            peak_allocated_bytes=policy.torch.cuda.max_memory_allocated(),
        )
    except BaseException as error:
        payload.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
        raise
    finally:
        save()
    print(
        {
            key: payload[key]
            for key in (
                "status",
                "canonical_history_ids_match",
                "groups_with_native_variation",
            )
        }
    )


if __name__ == "__main__":
    main()
