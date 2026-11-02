# Phase 5 three-seed reproducibility result

## Result

The decoder-aligned, full-group GRPO experiment reproduced across all three declared
training seeds.

| Training seed | Frozen pass@1 | Trained pass@1 | Trained greedy | First success group | First 8/8 group |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 2/50 (4%) | 50/50 (100%) | success | 3 | 8 |
| 47 | 2/50 (4%) | 50/50 (100%) | success | 1 | 6 |
| 71 | 2/50 (4%) | 50/50 (100%) | success | 0 | 3 |

Every seed improved by 96 percentage points under the canonical stochastic decoder.
Each trained policy's 50/50 Wilson 95% interval is 92.9%–100%; the frozen model's
2/50 interval is 1.1%–13.5%. All three trained greedy policies emitted the strict-valid
and successful `click('13')` action, while the frozen greedy policy failed.

The pooled count is 150/150 trained samples versus 6/150 frozen samples. A pooled
Wilson interval for trained sampling is 97.5%–100%, but pooling does not create 150
independent browser tasks: every evaluation uses the same seed-invariant prompt and the
same 50 generation seeds. The three per-training-seed results are the primary
reproducibility evidence.

## Learning trajectory across seeds

All runs used the same declared configuration:

- LFM2-350M with rank-8 LoRA;
- FP32 training on a Colab Tesla T4;
- 60 optimizer steps, each containing one full eight-completion GRPO group;
- per-device batch 8, generation batch 8, and `steps_per_generation=1`;
- temperature 1.0, top-p 0.95, no repetition penalty, and 32 completion tokens;
- formatting curriculum weight decaying to zero after group 40; and
- 50 paired stochastic evaluation samples plus one greedy evaluation.

Seed affected learning speed but not the final outcome:

| Seed | Strict successes in early groups | Groups with reward variance and positive gradients | Strict-phase success, groups 40–59 |
| ---: | --- | --- | ---: |
| 23 | 0, 0, 0, 2, 5, 6, 5, 7, 8 | 0–7 | 160/160 |
| 47 | 0, 2, 2, 2, 7, 7, 8 | 0–5 | 160/160 |
| 71 | 1, 3, 6, 8 | 0–2 | 160/160 |

Once a seed reached its first 8/8 group, every subsequent group remained perfect.
Across the three runs, all 480 strict-phase completions succeeded. All 164 trainable
LoRA tensors changed in every run, all reported training metrics were finite, and TRL's
policy-ratio clipping metrics remained zero under the one-group-per-step configuration.

The variation in first-success and saturation group demonstrates genuine training-seed
sensitivity in the path to the same saturated solution. Final entropy also differed,
but every trained sample was strict-valid, so that variation did not affect the task
metric.

## Runtime

| Seed | Training runtime | Peak allocated VRAM | Adapter SHA-256 |
| ---: | ---: | ---: | --- |
| 23 | 17m13s | 1.73 GiB | `268acb4c7459363d2d3f5812fb81fd797496c45c12f1b7104565434f3fc10ca8` |
| 47 | 17m16s | 1.72 GiB | `05f3c54c3a4d8228df19e8c9651578b8bd5bc2deaa79a13b4bb973c17d9abf7e` |
| 71 | 17m39s | 1.72 GiB | `aaf508b0784a79ecc7514096cafb098dec8cd9aba97ab51086f7ff41c6d39860` |

Total measured training time was 3,128.36 seconds, or 52 minutes 8 seconds. Mean
runtime was 17 minutes 23 seconds. All runs used a free Colab T4.

The runs were intentionally not shortened after seed 23 revealed early saturation.
Keeping the same 60-group declaration avoids post-result configuration changes in the
replication set. Most later groups have zero reward variance and zero gradients.

## Provenance

- PyTorch: 2.11.0+cu128
- Transformers: 4.57.3
- TRL: 0.29.0
- PEFT: 0.19.1
- BrowserGym: 0.14.3
- vLLM and OpenEnv: not installed

Each raw report embeds its Git commit, dependency versions, GPU, complete configuration,
per-completion rewards, trainer history, and evaluation episodes. Reports are in the
[evidence bundle](https://huggingface.co/datasets/Aptheos/lfm-browser-control-evidence).

## Scope

The three runs establish reproducible single-task stochastic and
greedy action-format alignment under the declared curriculum.

The supported claim is:

> Across three training seeds, full-group GRPO/LoRA with an explicit formatting
> curriculum and verifiable BrowserGym success reward aligned LFM2-350M to emit the
> strict `click('13')` action on the fixed MiniWoB `click-test` prompt.

The evidence does not establish held-out task generalization or isolate strict
environment reward from the answer-derived formatting curriculum. All policies
saturated while shaping remained active; the later strict phase verifies retention,
not strict-reward-driven acquisition. A strict-only ablation could address that causal
question.
