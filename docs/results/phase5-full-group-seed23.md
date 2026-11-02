# Phase 5 full-group GRPO training: seed 23

## Result

The decoder-aligned, full-group seed-23 run succeeded. Under the canonical stochastic
decoder used by GRPO:

| Policy | Strict successes | Pass@1 | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| Frozen | 2/50 | 4% | 1.1%–13.5% |
| Trained | 50/50 | 100% | 92.9%–100% |
| Difference | — | +96 percentage points | — |

Both frozen successes also succeeded under the trained policy. The other 48 paired
generation seeds succeeded only after training. An exploratory exact paired test gives
a two-sided `p ≈ 7.1 × 10⁻¹⁵`.

The trained greedy output was also strict-valid and successful:
`click('13')`. The frozen greedy output remained malformed HTML. Unlike the earlier
pilot, learning therefore moved the correct action into both the stochastic
distribution and its greedy mode.

These measurements concern one seed-invariant `click-test` prompt. They demonstrate
single-task action-format alignment, not held-out browser-task generalization.

## Training trajectory

The corrected batching processed one complete eight-completion GRPO group per optimizer
step:

| Setting | Value |
| --- | ---: |
| Optimizer steps / groups | 60 |
| Generations per group | 8 |
| Total completions | 480 |
| Per-device training batch | 8 |
| Generation batch | 8 |
| `steps_per_generation` | 1 |
| Gradient accumulation | 1 |

This matches the pinned TRL 0.29 contract and avoids reusing one generation group over
four optimizer steps, as happened in the first controlled pilot.

Learning was rapid:

| Group | Curriculum weight | Strict successes |
| ---: | ---: | ---: |
| 0–2 | 1.000–0.950 | 0/24 |
| 3 | 0.925 | 2/8 |
| 4 | 0.900 | 5/8 |
| 5 | 0.875 | 6/8 |
| 6 | 0.850 | 5/8 |
| 7 | 0.825 | 7/8 |
| 8 | 0.800 | 8/8 |
| 9–39 | 0.775–0.025 | 248/248 |
| 40–59 | 0.000 | 160/160 |

Reward variance and positive gradient norms occurred only in groups 0–7. Group 8 was
the first fully successful group; every later group also scored 8/8, including all 20
strict-reward groups after curriculum weight reached zero. All 164 trainable LoRA
tensors changed.

The logged scalar loss rounds to zero after the first step despite positive gradients
through step 8. This is compatible with the group-normalized on-policy objective: the
reported mean surrogate can be near zero while individual token contributions still
produce gradients. TRL's policy clip-ratio metrics stayed zero. The separate
`completions/clipped_ratio` was 0.25 only at step 1 and zero thereafter.

## Saturation and compute efficiency

The run took 1,033.48 seconds, or 17 minutes 13 seconds, for training. It reached full
group success after approximately eight groups, but then spent 52 more groups with zero
reward variance and zero gradients. About 87% of the declared optimizer steps therefore
provided no additional GRPO learning signal.

This does not invalidate the result: the predeclared 60-group run verified that the
policy retained perfect sampled success through 20 groups of strict environment reward.
It does show that most of the later groups added nothing. The seed-47 and seed-71 runs
kept the exact configuration so that their outcomes stayed comparable.

## Runtime and provenance

- Adapter SHA-256:
  `268acb4c7459363d2d3f5812fb81fd797496c45c12f1b7104565434f3fc10ca8`
- GPU: Tesla T4, compute capability 7.5
- PyTorch: 2.11.0+cu128
- Transformers: 4.57.3
- TRL: 0.29.0
- PEFT: 0.19.1
- BrowserGym: 0.14.3
- Precision: FP32
- Peak allocated VRAM: 1,862,420,480 bytes, approximately 1.73 GiB
- Training loss: -0.00500538
- Finite training metrics: yes
- vLLM and OpenEnv: not installed

The raw report is in the
[evidence bundle](https://huggingface.co/datasets/Aptheos/lfm-browser-control-evidence). It embeds the complete manifest,
per-completion rewards, trainer history, and frozen/trained episode telemetry.

## Interpretation

The result supports the narrow claim:

> With an explicit formatting curriculum and verifiable BrowserGym success reward,
> full-group GRPO/LoRA training aligned LFM2-350M to emit the strict `click('13')`
> action on the fixed MiniWoB `click-test` prompt.

It does not isolate the effect of strict environment reward from the answer-derived
formatting curriculum. The first successful samples appeared while curriculum weight
was 0.925, and the policy saturated before the curriculum reached zero. A future
strict-reward-only experiment would be required to make a stronger causal claim.
