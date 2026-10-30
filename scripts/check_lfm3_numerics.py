"""Compare FP16/T4 and CPU BF16 logits on development prompts; no training."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from browser_control.book_flight import BOOK_FLIGHT_TOOLS
from browser_control.lfm3 import MODEL_ID, MODEL_REVISION, LFM3Policy
from browser_control.transformers_policy import DecoderConfig


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = [
        record
        for record in json.loads(args.records.read_text())
        if record["task"] == "book-flight-nodelay" and record["task_seed"] in (0, 1)
    ]
    assert len(records) == 2
    policy = LFM3Policy.load(DecoderConfig("greedy", False, 128), "native-tools")
    assert next(policy.model.parameters()).dtype == torch.float16, (
        "requires an FP16 device such as T4"
    )
    policy.tools = BOOK_FLIGHT_TOOLS
    print("Loading pinned CPU BF16 reference", flush=True)
    cpu_model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to("cpu")
        .eval()
    )
    report = {
        "model_revision": MODEL_REVISION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_dtype": "float16",
        "cpu_dtype": "bfloat16",
        "comparisons": [],
        "scope": "two initial development prompts only; no gradient or long-context equivalence claim",
    }
    for record in records:
        messages = record["steps"][0]["messages"]
        inputs = policy.encode(messages)
        started = time.monotonic()
        with torch.inference_mode():
            gpu = (
                policy.model(**inputs, logits_to_keep=1).logits[:, -1, :].float().cpu()
            )
            cpu = (
                cpu_model(
                    **{name: tensor.cpu() for name, tensor in inputs.items()},
                    logits_to_keep=1,
                )
                .logits[:, -1, :]
                .float()
            )
        assert torch.isfinite(cpu).all() and torch.isfinite(gpu).all(), (
            "non-finite logits"
        )
        difference = (gpu - cpu).abs()
        cpu_logp, gpu_logp = cpu.log_softmax(-1), gpu.log_softmax(-1)
        entry = {
            "seed": record["task_seed"],
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "finite": True,
            "mean_absolute_difference": difference.mean().item(),
            "max_absolute_difference": difference.max().item(),
            "top1_agrees": bool(cpu.argmax(-1).item() == gpu.argmax(-1).item()),
            "kl_cpu_to_gpu": (cpu_logp.exp() * (cpu_logp - gpu_logp)).sum().item(),
            "seconds": time.monotonic() - started,
        }
        report["comparisons"].append(entry)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
    report.update(
        status="complete",
        all_finite=True,
        all_top1_agree=all(row["top1_agrees"] for row in report["comparisons"]),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
