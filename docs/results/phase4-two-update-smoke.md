# Phase 4 two-update smoke result

Date: 2026-07-21  
Colab notebook commit: `33b81c7` — executed two-update run with zero reward variance
and zero gradients

Training-code commit: `c0dee8a` — idempotent Colab repository setup and smoke-run
implementation used for this attempt

Raw local artifact: `smoke-report.json` (ignored)  
Adapter archive: `phase4-smoke-adapter.zip` (ignored)

## Runtime

- GPU: Tesla T4, compute capability 7.5
- GPU memory: 14.56 GiB
- PyTorch: 2.11.0+cu128
- Training dtype: FP16
- Peak allocated GPU memory: 788,905,984 bytes (0.735 GiB)
- TRL contract tests: 3 passed

Colab's preinstalled `torchao==0.10.0` was incompatible with PEFT 0.19.1.
Upgrading to 0.17.0 allowed LoRA injection but emitted failures while loading two
unused native extensions. The maintained notebook now uninstalls TorchAO because
this experiment does not use quantization or TorchAO kernels.

## Mechanical smoke result

- Global optimizer steps: 2
- Trainable parameter tensors: 164, all LoRA-owned
- All seven validated LFM2 target suffixes were attached
- Training runtime: 4.66 seconds
- Adapter save and reload: succeeded
- Compressed adapter archive: 3.44 MiB
- Adapter weights inside archive: 10,703,800 bytes

## Learning signal

- Mean reward: 0.0
- Reward standard deviation: 0.0
- Fraction of zero-standard-deviation groups: 1.0
- Loss: 0.0 on both steps
- Gradient norm: 0.0 on both steps
- Truncated completion fraction: 0.5

The run therefore validates the GPU, trainer, LoRA attachment, BrowserGym reward,
checkpoint, and reload wiring, but it did not perform a nonzero policy update.
Every sampled group had identical zero reward, so GRPO correctly produced no
gradient.

## Reload evaluation

The reloaded adapter scored 0/2 with a 100% invalid-action rate. Both outputs were:

```text
click(<button id="13">Click Me!</button>)
```

This matches the frozen baseline failure mode and is expected because the smoke
run had no learning signal. Phase 4 remains in progress until a second smoke run
produces reward diversity, finite nonzero gradients, and a confirmed LoRA update.

## Nonzero-update rerun

- Date: 2026-07-21
- Colab notebook commit: `e844418` — successful executed FP32 run with finite nonzero
  gradients and changed LoRA tensors
- Training-code commit: `36da0ad` — FP32 nonzero-update configuration used by the
  successful rerun
- Raw local artifact: `smoke-report (2).json` (ignored)
- Adapter archive: `phase4-smoke-nonzero-adapter.zip` (ignored)

The second run used the temporary formatting curriculum and FP32 after the first
curriculum attempt produced NaN FP16 gradient norms.

- Global optimizer steps: 2
- Reward mean: 0.0630
- Reward standard deviation: 0.03484
- Fraction of zero-standard-deviation groups: 0.0
- Losses: 0.6836 and -0.7450
- Gradient norms: 5.1442 and 7.8142
- LoRA tensors trainable: 164
- LoRA tensors changed: 164
- Training runtime: 5.67 seconds
- Peak allocated GPU memory: 1,514,244,096 bytes (1.410 GiB)
- Adapter save and reload: succeeded

The four sampled curriculum rewards were distinct enough to produce a real update,
but none of the completions passed the strict action parser. The two-seed reloaded
evaluation remained 0/2 with the original malformed-action failure. This is not
evidence of task learning; it closes Phase 4 because the complete optimizer and
checkpoint path now has a verified nonzero update. Phase 5 must train for longer
and evaluate only with strict environment-success reward.
