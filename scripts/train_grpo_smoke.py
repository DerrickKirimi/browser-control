"""Run exactly two LoRA GRPO updates on local BrowserGym click-test."""

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
    SmokeTrainingConfig,
    configure_model_for_training,
    create_lora_config,
    trainable_parameter_names,
    validate_lora_targets,
)
from browser_control.transformers_policy import TransformersPolicy


def _assert_finite_metrics(history: list[dict[str, object]]) -> None:
    checked = 0
    for record in history:
        for key in ("loss", "grad_norm", "train_loss"):
            value = record.get(key)
            if isinstance(value, int | float):
                checked += 1
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"non-finite {key}: {value}")
    if checked == 0:
        raise RuntimeError("trainer emitted no loss or gradient metrics")


def _has_positive_metric(history: list[dict[str, object]], key: str) -> bool:
    return any(
        isinstance(record.get(key), int | float) and float(record[key]) > 0.0
        for record in history
    )


def run(output_dir: Path, assets_dir: Path) -> dict[str, object]:
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

    config = SmokeTrainingConfig()
    device = inspect_cuda_device(torch)
    dtype = getattr(torch, device.training_dtype)
    document_root = ensure_miniwob_assets(assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with MiniwobAssetServer(document_root) as server:
        def environment_factory() -> LocalBrowserGymEnvironment:
            return LocalBrowserGymEnvironment.miniwob(
                "click-test", miniwob_url=server.base_url
            )

        rows = build_click_test_rows(environment_factory, (23, 24))
        dataset = Dataset.from_list(rows)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        model = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=dtype)
        configure_model_for_training(model)
        target_counts = validate_lora_targets(name for name, _ in model.named_modules())

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
            fp16=device.training_dtype == "float16",
            bf16=device.training_dtype == "bfloat16",
            gradient_checkpointing=True,
            logging_steps=1,
            logging_first_step=True,
            save_strategy="no",
            report_to="none",
            seed=config.seed,
            data_seed=config.seed,
            remove_unused_columns=False,
        )
        reward_function = make_click_test_reward(
            environment_factory, format_curriculum=True
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
        if trainer.state.global_step != config.max_steps:
            raise RuntimeError(
                f"expected {config.max_steps} updates, got {trainer.state.global_step}"
            )
        _assert_finite_metrics(trainer.state.log_history)
        changed_parameters = tuple(
            name
            for name, parameter in trainer.model.named_parameters()
            if name in before_update
            and not torch.equal(before_update[name], parameter.detach().cpu())
        )
        reward_values = {record["reward"] for record in reward_function.records}
        reward_diversity = len(reward_values) > 1
        nonzero_gradient = _has_positive_metric(
            trainer.state.log_history, "grad_norm"
        )
        adapter_dir = output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        global_step = trainer.state.global_step
        log_history = trainer.state.log_history
        training_loss = result.training_loss

        del trainer, model, result
        gc.collect()
        torch.cuda.empty_cache()

        reloaded_base = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=dtype)
        reloaded_model = PeftModel.from_pretrained(reloaded_base, adapter_dir)
        reloaded_model.to(f"cuda:{device.index}")
        reloaded_model.eval()
        reloaded_policy = TransformersPolicy(
            model=reloaded_model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
        )
        reload_evaluation = evaluate_click_test(
            environment_factory, reloaded_policy, seeds=(43, 44)
        )

    report = {
        "model_id": config.model_id,
        "global_steps": global_step,
        "training_loss": training_loss,
        "trainable_parameter_tensors": len(trainable),
        "changed_parameter_tensors": len(changed_parameters),
        "lora_target_counts": target_counts,
        "dtype": device.training_dtype,
        "gpu": device.name,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device.index)),
        "log_history": log_history,
        "format_curriculum": True,
        "reward_diversity": reward_diversity,
        "reward_records": reward_function.records,
        "nonzero_gradient": nonzero_gradient,
        "reload_evaluation": reload_evaluation.to_dict(),
    }
    if not math.isfinite(float(training_loss)):
        raise RuntimeError(f"non-finite final training loss: {training_loss}")
    report_path = output_dir / "smoke-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    failures: list[str] = []
    if not reward_diversity:
        failures.append("sampled rewards had no diversity")
    if not nonzero_gradient:
        failures.append("trainer emitted no positive gradient norm")
    if not changed_parameters:
        failures.append("no LoRA parameter tensor changed")
    if failures:
        raise RuntimeError(
            "; ".join(failures) + f"; inspect the preserved report at {report_path}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase4-smoke"))
    parser.add_argument(
        "--assets-dir", type=Path, default=Path("artifacts/miniwob-plusplus")
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.assets_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
