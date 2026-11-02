"""Real TRL1.16 image/ID/mask updates on tiny random LFM3-VL CPU weights.

Actions and native outcomes are scripted. Images use reduced resolution for this
architecture/batching fixture. This is not pretrained/browser/T4 validation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path
from types import MethodType

from check_lfm3_vision_contract import tiny_vision_model

from browser_control.book_flight_rollout import native_success_reward
from browser_control.lfm3_vision import (
    COORDINATE_TOOLS,
    MODEL_REVISION,
    build_vision_messages,
)
from browser_control.vision_browser import VisionObservation
from browser_control.vision_trainer import (
    StagedVisionRollout,
    native_vision_grpo_trainer_class,
    validate_visual_trainer_rows,
)


def main():
    import torch
    import trl
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--constant-rewards", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier contract evidence")
    if version("transformers") != "5.21.0" or trl.__version__ != "1.16.0":
        parser.error("This contract pins Transformers5.21.0/TRL1.16.0")
    torch.set_num_threads(2)
    torch.manual_seed(23)
    processor = AutoProcessor.from_pretrained(
        args.model_source, trust_remote_code=True, local_files_only=True
    )
    processor.image_processor.size = {"shortest_edge": 1024, "longest_edge": 2048}
    model = get_peft_model(
        tiny_vision_model(args.model_source),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)$",
            task_type="CAUSAL_LM",
        ),
    )
    if any(
        "lora_" not in name or "language_model" not in name
        for name, p in model.named_parameters()
        if p.requires_grad
    ):
        raise ValueError("Only language-model LoRA may train")

    def collect(inputs, trainer):
        rows = []
        tokenizer = processor.tokenizer
        for index, item in enumerate(inputs):
            seed = json.loads(item["prompt"])["task_seed"]
            first = Image.new("RGB", (32, 32), "red" if seed == 100 else "blue")
            second = Image.new("RGB", (64, 32), "green" if seed == 100 else "yellow")
            png = io.BytesIO()
            first.save(png, format="PNG")
            obs = VisionObservation(f"scripted fixture{seed}", png.getvalue(), 32, 32)
            encoded = processor.apply_chat_template(
                build_vision_messages(obs),
                tools=COORDINATE_TOOLS,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            future = processor.image_processor(images=[second], return_tensors="pt")
            future_tokens = (
                int(future["image_grid_thw"].prod(-1).sum())
                // processor.image_processor.merge_size**2
            )
            native = args.constant_rewards or index % 2 == 0
            generated_first = tokenizer.encode("continue", add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            feedback = (
                tokenizer.encode("\nobservation", add_special_tokens=False)
                + [125009]
                + [125030] * future_tokens
                + [125010]
            )
            generated_second = tokenizer.encode(
                "left" if native else "rightmost edge", add_special_tokens=False
            ) + [tokenizer.eos_token_id]
            completion = generated_first + feedback + generated_second
            mask = (
                [1] * len(generated_first)
                + [0] * len(feedback)
                + [1] * len(generated_second)
            )
            rows.append(
                {
                    "prompt_ids": encoded["input_ids"][0].tolist(),
                    "completion_ids": completion,
                    "env_mask": mask,
                    "logprobs": [-2.0 if value else 0.0 for value in mask],
                    "native_success": native,
                    "images": [first, second],
                    "processing_messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "image"},
                                {"type": "text", "text": obs.goal},
                            ],
                        }
                    ],
                }
            )
        return rows

    # The future image sits in masked external completion context, not the initial
    # prompt. Its features must not affect an earlier action's causal logits.
    sample = collect([{"prompt": '{"task_seed":100}'}], None)[0]
    images = processor.image_processor(images=sample["images"], return_tensors="pt")
    full_ids = torch.tensor([sample["prompt_ids"] + sample["completion_ids"]])
    positions = torch.tensor([len(sample["prompt_ids"]) - 1, full_ids.shape[1] - 2])
    first_patches = int(images["image_grid_thw"][0].prod())
    altered_pixels = images["pixel_values"].clone()
    altered_pixels[first_patches:] = 0
    with torch.inference_mode():
        original = model(
            input_ids=full_ids, **images, use_cache=False, logits_to_keep=positions
        ).logits
        altered = model(
            input_ids=full_ids,
            **{**images, "pixel_values": altered_pixels},
            use_cache=False,
            logits_to_keep=positions,
        ).logits
    early_delta = float((original[:, 0] - altered[:, 0]).abs().max())
    late_delta = float((original[:, 1] - altered[:, 1]).abs().max())
    if early_delta > 1e-6 or late_delta <= 1e-7:
        raise ValueError("Future image causal isolation/influence check failed")

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    trainable_count = sum(p.requires_grad for p in model.parameters())
    rollout = StagedVisionRollout(collect)
    config = trl.GRPOConfig(
        output_dir=str(args.output.parent / (args.output.stem + "-trainer")),
        use_cpu=True,
        bf16=False,
        fp16=False,
        use_vllm=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        steps_per_generation=4,
        num_iterations=1,
        max_steps=2,
        max_completion_length=64,
        learning_rate=1e-3,
        weight_decay=0.0,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        mask_truncated_completions=False,
        disable_dropout=True,
        gradient_checkpointing=False,
        temperature=1.0,
        top_p=0.95,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        disable_tqdm=True,
        seed=23,
    )
    trainer = native_vision_grpo_trainer_class()(
        model=model,
        args=config,
        processing_class=processor,
        reward_funcs=native_success_reward,
        train_dataset=Dataset.from_list(
            [{"prompt": json.dumps({"task_seed": seed})} for seed in (100, 101)]
        ),
        rollout_func=rollout,
    )
    checks, rejections = [], []
    original_loss, original_logps = (
        trainer._compute_loss,
        trainer._get_per_token_logps_and_entropies,
    )

    def checked_loss(self, current_model, inputs):
        self._fixture_active = inputs["completion_mask"] * inputs["tool_mask"]
        if not checks:
            for field in (
                "pixel_values",
                "image_grid_thw",
                "advantages",
                "tool_mask",
                "completion_ids",
            ):
                corrupted = dict(inputs)
                corrupted[field] = inputs[field].clone()
                if field == "tool_mask":
                    zero = (inputs[field][0] == 0).nonzero()[0].item()
                    corrupted[field][0, zero] = 1
                else:
                    corrupted[field].view(-1)[0] += 1
                try:
                    validate_visual_trainer_rows(corrupted, rollout.episodes)
                except ValueError:
                    rejections.append(field)
                else:
                    raise ValueError(
                        "Corrupted visual trainer input accepted: " + field
                    )
        checks.append(
            {
                "raw_ids_preserved": True,
                "masked_context_tokens": int((self._fixture_active == 0).sum()),
            }
        )
        return original_loss(current_model, inputs)

    def checked_logps(self, *positional, **kwargs):
        result = original_logps(*positional, **kwargs)
        if result[0].requires_grad:
            active = self._fixture_active
            record = checks[-1]

            def gradient_check(gradient):
                if (
                    not torch.isfinite(gradient).all()
                    or torch.any(gradient[active == 0] != 0)
                    or bool(torch.any(gradient[active == 1] != 0))
                    == args.constant_rewards
                ):
                    raise ValueError(
                        "Visual policy gradient mask/finiteness check failed"
                    )
                record["external_gradients_zero"] = True

            result[0].register_hook(gradient_check)
        return result

    trainer._compute_loss = MethodType(checked_loss, trainer)
    trainer._get_per_token_logps_and_entropies = MethodType(checked_logps, trainer)
    trainer.train()
    changed = [
        name
        for name, p in model.named_parameters()
        if not torch.equal(p.detach(), before[name])
    ]
    if (
        trainer.state.global_step != 2
        or len(checks) != 8
        or len(trainer.vision_binding_checks) != 8
        or len(trainer.vision_loss_checks) != 8
        or len(rollout.episodes) != 8
        or bool(changed) == args.constant_rewards
        or (not args.constant_rewards and len(changed) != trainable_count)
        or any("lora_" not in name or "language_model" not in name for name in changed)
        or any(not torch.isfinite(p).all() for p in model.parameters())
        or not all(row.get("external_gradients_zero") for row in checks)
        or not all(
            row.get("external_gradients_zero") for row in trainer.vision_loss_checks
        )
    ):
        raise ValueError("Multimodal trainer weights/steps/image/mask contract failed")
    report = {
        "status": "passed",
        "scope": "tiny random CPU model, reduced-resolution images, scripted actions/outcomes; not pretrained/browser/T4 training or real sampling",
        "model_revision": MODEL_REVISION,
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "trl",
                "peft",
                "accelerate",
                "datasets",
                "fsspec",
                "huggingface-hub",
                "tokenizers",
                "safetensors",
                "numpy",
                "pillow",
            )
        },
        "source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()
            ).hexdigest()
            for name in (
                "scripts/check_lfm3_vision_trl_contract.py",
                "scripts/check_lfm3_vision_contract.py",
                "src/browser_control/vision_trainer.py",
                "src/browser_control/lfm3_trainer.py",
            )
        },
        "optimizer_steps": trainer.state.global_step,
        "constant_rewards": args.constant_rewards,
        "episodes": len(rollout.episodes),
        "changed_adapter_tensors": len(changed),
        "trainable_adapter_tensors": trainable_count,
        "frozen_base_unchanged": True,
        "image_bindings": trainer.vision_binding_checks,
        "loss_checks": checks,
        "production_loss_checks": trainer.vision_loss_checks,
        "future_image_early_logit_delta": early_delta,
        "future_image_late_logit_delta": late_delta,
        "rejected_corruptions": rejections,
        "image_processor_size_override": processor.image_processor.size,
        "model_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.model_source.iterdir())
            if path.is_file()
        },
        "prompt_image_tokens": [
            row["prompt_ids"].count(125030) for row in rollout.episodes
        ],
        "external_image_tokens": [
            sum(
                token == 125030 and not mask
                for token, mask in zip(row["completion_ids"], row["env_mask"])
            )
            for row in rollout.episodes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        {
            key: report[key]
            for key in (
                "status",
                "optimizer_steps",
                "changed_adapter_tensors",
                "future_image_early_logit_delta",
                "future_image_late_logit_delta",
            )
        }
    )


if __name__ == "__main__":
    main()
