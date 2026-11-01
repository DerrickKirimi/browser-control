"""CPU vision processor/architecture check using tiny random float32 weights.

No pretrained weight download, GPU execution or GRPO performance claim. Download
the model's config, tokenizer, processor, template and Python files at the pinned
revision into --model-source before running this offline check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from browser_control.lfm3_vision import (
    COORDINATE_TOOLS,
    MODEL_ID,
    MODEL_REVISION,
    LFM3VisionPolicy,
    build_vision_messages,
)
from browser_control.transformers_policy import DecoderConfig
from browser_control.vision_browser import VisionObservation


def tiny_vision_model(model_source):
    """Construct a small random instance of the pinned multimodal architecture."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(
        model_source, trust_remote_code=True, local_files_only=True
    )
    text = config.text_config.to_dict()
    text.update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["conv", "full_attention", "sliding_attention"],
        max_position_embeddings=4096,
        sliding_window=16,
        dtype="float32",
    )
    vision = config.vision_config.to_dict()
    vision.update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention", "conv"],
        dtype="float32",
    )
    tiny = type(config)(
        text_config=text,
        vision_config=vision,
        projector_hidden_size=64,
        dtype="float32",
        auto_map=config.auto_map,
    )
    tiny._name_or_path = str(model_source)
    return AutoModelForImageTextToText.from_config(
        tiny, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier contract evidence")
    if version("transformers") != "5.21.0":
        parser.error("This CPU contract pins Transformers5.21.0")
    torch.set_num_threads(2)
    torch.manual_seed(23)
    image = Image.open(args.screenshot)
    observation = VisionObservation(
        args.goal, args.screenshot.read_bytes(), image.width, image.height
    )
    processor = AutoProcessor.from_pretrained(
        args.model_source, trust_remote_code=True, local_files_only=True
    )
    inputs = processor.apply_chat_template(
        build_vision_messages(observation),
        tools=COORDINATE_TOOLS,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    model = tiny_vision_model(args.model_source)
    config = model.config
    pixels = inputs["pixel_values"].detach().float().requires_grad_(True)
    inputs["pixel_values"] = pixels
    output = model(**inputs, use_cache=False, logits_to_keep=1)
    logits = output.logits
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite tiny vision logits")
    target = processor.tokenizer.encode("click", add_special_tokens=False)[0]
    loss = -logits[0, -1].log_softmax(-1)[target]
    loss.backward()
    if (
        pixels.grad is None
        or not torch.isfinite(pixels.grad).all()
        or not torch.any(pixels.grad != 0)
    ):
        raise ValueError("Image input did not receive a finite nonzero gradient")
    if any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    ):
        raise ValueError("Nonfinite model gradient")
    with torch.inference_mode():
        blank_logits = model(
            **{**inputs, "pixel_values": torch.zeros_like(pixels)},
            use_cache=False,
            logits_to_keep=1,
        ).logits
    difference = float((logits.detach() - blank_logits).abs().max())
    if not torch.isfinite(blank_logits).all() or difference <= 0:
        raise ValueError("Image ablation did not change finite logits")
    model.generation_config.top_k = 50
    model.generation_config.repetition_penalty = 1.1
    policy = LFM3VisionPolicy(
        model, processor, torch, DecoderConfig("tiny-greedy", False, 2), device="cpu"
    )
    numerical = policy.check_logits(observation)
    generated = policy.generate(observation, seed=23)
    repeated = policy.generate(observation, seed=47)
    if (
        model.generation_config.top_k != 0
        or model.generation_config.repetition_penalty != 1.0
        or any(p.requires_grad for p in model.parameters())
        or generated.prompt_token_ids != tuple(inputs["input_ids"][0].tolist())
        or not 0 < generated.completion_tokens <= 2
        or generated.completion_token_ids != repeated.completion_token_ids
    ):
        raise ValueError("Frozen bounded vision generation contract failed")

    def nonfinite_output(_module, _arguments, output):
        output.logits.fill_(float("nan"))

    corruption = model.register_forward_hook(nonfinite_output)
    try:
        try:
            policy.generate(observation, seed=23)
        except RuntimeError as error:
            if "Nonfinite vision generation logits" not in str(error):
                raise
        else:
            raise ValueError("Nonfinite generation logits were accepted")
    finally:
        corruption.remove()
    report = {
        "status": "passed",
        "scope": "CPU tiny random architecture/processor forward/backward and bounded generation; no pretrained/GPU/GRPO result",
        "model_id": MODEL_ID,
        "intended_model_revision": MODEL_REVISION,
        "packages": {
            name: version(name)
            for name in ("torch", "torchvision", "transformers", "huggingface-hub")
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.model_source.iterdir())
            if path.is_file()
        },
        "project_source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()
            ).hexdigest()
            for name in (
                "scripts/check_lfm3_vision_contract.py",
                "src/browser_control/lfm3_vision.py",
                "src/browser_control/vision_browser.py",
                "src/browser_control/transformers_policy.py",
            )
        },
        "screenshot_sha256": hashlib.sha256(observation.screenshot_png).hexdigest(),
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "image_tokens": int((inputs["input_ids"] == config.image_token_id).sum()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss.detach()),
        "pixel_gradient_norm": float(pixels.grad.norm()),
        "blank_image_max_logit_difference": difference,
        "pretrained_weights_loaded": False,
        "frozen_inference": {
            "numerical_check": numerical,
            "prompt_ids_preserved": True,
            "completion_token_ids": list(generated.completion_token_ids),
            "completion_cap": 2,
            "greedy_ids_match_across_rng_seeds": True,
            "top_k": model.generation_config.top_k,
            "repetition_penalty": model.generation_config.repetition_penalty,
            "all_weights_frozen": True,
            "generation_logits_checks": generated.generation_logits_checks,
            "nonfinite_generation_rejected": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        {
            key: report[key]
            for key in (
                "status",
                "parameters",
                "image_tokens",
                "pixel_gradient_norm",
                "blank_image_max_logit_difference",
            )
        }
    )


if __name__ == "__main__":
    main()
