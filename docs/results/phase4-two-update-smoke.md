# Phase 4 two-update smoke result

Date: 2026-07-21  
Colab notebook commit: `33b81c7`  
Training-code commit: `c0dee8a`  
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
