"""Run one predeclared Phase 5 train/evaluate comparison on a Colab GPU."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

from browser_control.evaluation import evaluate_click_test
from browser_control.hardware import inspect_cuda_device
from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.rewards import build_click_test_rows, make_click_test_reward
from browser_control.training import (
    ControlledTrainingConfig,
    configure_model_for_training,
    create_lora_config,
    trainable_parameter_names,
    validate_lora_targets,
)
from browser_control.transformers_policy import TransformersPolicy

EVALUATION_SEEDS = tuple(range(1000, 1020))
TRAIN_EPISODES = 20


def _finite_training_metrics(history: list[dict[str, object]]) -> bool:
    values = [
        float(record[key])
        for record in history
        for key in ("loss", "grad_norm", "train_loss")
        if isinstance(record.get(key), int | float)
    ]
    return bool(values) and all(math.isfinite(value) for value in values)


def run(
    *,
    seed: int,
    max_steps: int,
    curriculum_steps: int,
    output_dir: Path,
    assets_dir: Path,
) -> dict[str, object]:
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install the environment, model, and training extras before this run"
        ) from error

    config = ControlledTrainingConfig(
        seed=seed,
        max_steps=max_steps,
        curriculum_steps=curriculum_steps,
    )
    device = inspect_cuda_device(torch)
    dtype = torch.float32
    document_root = ensure_miniwob_assets(assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run-status.json"
    run_stem = f"phase5-click-test-seed-{seed:04d}-steps-{max_steps:04d}"
    report_path = output_dir / f"{run_stem}-report.json"
    status_path.write_text(
        json.dumps({"status": "running", "report": report_path.name}, indent=2)
        + "\n"
    )

    with MiniwobAssetServer(document_root) as server:
        def environment_factory() -> LocalBrowserGymEnvironment:
            return LocalBrowserGymEnvironment.miniwob(
                "click-test", miniwob_url=server.base_url
            )

        train_seeds = tuple(range(seed, seed + TRAIN_EPISODES))
        rows = build_click_test_rows(environment_factory, train_seeds)
        dataset = Dataset.from_list(rows)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        model = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=dtype)
        model.to(f"cuda:{device.index}")
        model.eval()
        frozen_policy = TransformersPolicy(model, tokenizer, torch, device)
        frozen_evaluation = evaluate_click_test(
            environment_factory, frozen_policy, EVALUATION_SEEDS
        )

        configure_model_for_training(model)
        target_counts = validate_lora_targets(name for name, _ in model.named_modules())
        reward_function = make_click_test_reward(
            environment_factory,
            format_curriculum=True,
            curriculum_steps=config.curriculum_steps,
        )
        arguments = GRPOConfig(
            output_dir=str(output_dir),
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=1,
            generation_batch_size=config.num_generations,
            num_generations=config.num_generations,
            max_completion_length=config.max_completion_length,
            temperature=1.0,
            top_p=0.95,
            use_vllm=False,
            beta=0.0,
            num_iterations=1,
            epsilon=0.2,
            importance_sampling_level="token",
            scale_rewards="group",
            loss_type="grpo",
            mask_truncated_completions=True,
            fp16=False,
            bf16=False,
            gradient_checkpointing=True,
            logging_steps=1,
            logging_first_step=True,
            save_strategy="no",
            report_to="none",
            seed=config.seed,
            data_seed=config.seed,
            shuffle_dataset=False,
            remove_unused_columns=False,
        )
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_function,
            train_dataset=dataset,
            args=arguments,
            peft_config=create_lora_config(config),
        )
        trainable = trainable_parameter_names(trainer.model.named_parameters())
        before_update = {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainer.model.named_parameters()
            if name in trainable
        }
        result = trainer.train()
        changed = tuple(
            name
            for name, parameter in trainer.model.named_parameters()
            if name in before_update
            and not torch.equal(before_update[name], parameter.detach().cpu())
        )
        adapter_dir = output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        log_history = trainer.state.log_history
        global_step = trainer.state.global_step
        training_loss = result.training_loss

        preliminary = {
            "seed": seed,
            "train_seeds": train_seeds,
            "evaluation_seeds": EVALUATION_SEEDS,
            "max_steps": config.max_steps,
            "curriculum_steps": config.curriculum_steps,
            "global_steps": global_step,
            "dtype": "float32",
            "gpu": device.name,
            "model_id": config.model_id,
            "training_loss": training_loss,
            "finite_training_metrics": _finite_training_metrics(log_history),
            "trainable_parameter_tensors": len(trainable),
            "changed_parameter_tensors": len(changed),
            "lora_target_counts": target_counts,
            "reward_records": reward_function.records,
            "log_history": log_history,
            "frozen_evaluation": frozen_evaluation.to_dict(),
            "trained_evaluation": None,
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device.index)
            ),
        }
        report_path.write_text(
            json.dumps(preliminary, indent=2, sort_keys=True) + "\n"
        )
        if global_step != config.max_steps:
            raise RuntimeError(f"expected {config.max_steps} steps, got {global_step}")
        if not preliminary["finite_training_metrics"]:
            raise RuntimeError("training emitted non-finite metrics")
        if not changed:
            raise RuntimeError("no LoRA parameter changed")

        del trainer, model, result
        gc.collect()
        torch.cuda.empty_cache()

        reloaded_base = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=dtype)
        reloaded_model = PeftModel.from_pretrained(reloaded_base, adapter_dir)
        reloaded_model.to(f"cuda:{device.index}")
        reloaded_model.eval()
        trained_policy = TransformersPolicy(reloaded_model, tokenizer, torch, device)
        trained_evaluation = evaluate_click_test(
            environment_factory, trained_policy, EVALUATION_SEEDS
        )

    preliminary["trained_evaluation"] = trained_evaluation.to_dict()
    preliminary["strict_success_delta"] = (
        trained_evaluation.success_rate - frozen_evaluation.success_rate
    )
    report_path.write_text(json.dumps(preliminary, indent=2, sort_keys=True) + "\n")
    status_path.write_text(
        json.dumps({"status": "success", "report": report_path.name}, indent=2)
        + "\n"
    )
    return preliminary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--curriculum-steps", type=int, default=40)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/phase5/seed-23")
    )
    parser.add_argument(
        "--assets-dir", type=Path, default=Path("artifacts/miniwob-plusplus")
    )
    args = parser.parse_args()
    report = run(
        seed=args.seed,
        max_steps=args.max_steps,
        curriculum_steps=args.curriculum_steps,
        output_dir=args.output_dir,
        assets_dir=args.assets_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
