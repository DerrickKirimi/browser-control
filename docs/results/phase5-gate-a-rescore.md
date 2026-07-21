# Phase 5 Gate A: paired checkpoint re-score

## Result

Gate A overturns the original negative interpretation. Under the decoder GRPO
actually trained with:

| Policy | Successes | Pass@1 | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| Frozen | 2/50 | 4% | 1.1%–13.5% |
| Trained | 16/50 | 32% | 20.8%–45.8% |
| Difference | — | +28 percentage points | — |

Both frozen successes also occurred under the trained policy. Fourteen paired
generation seeds succeeded only under the trained policy, while none succeeded only
under the frozen policy. An exploratory exact paired test gives a two-sided
`p ≈ 0.00012`.

This statistic describes stochastic outputs for one fixed prompt. It does not measure
generalization across browser tasks, and the paired test should not be interpreted as
evidence over independently sampled task instances.

## Decoder-mismatch finding

The repetition-penalty hypothesis is directly supported:

| Decoder | Frozen | Trained |
| --- | ---: | ---: |
| Existing evaluation, repetition penalty 1.05 | 0/50 | 0/50 |
| Same evaluation settings, repetition penalty 1.0 | 0/50 | 12/50 |
| Training decoder | 2/50 | 16/50 |
| Greedy | 0/1 | 0/1 |

Changing only repetition penalty from 1.05 to 1.0 recovered 12 trained successes,
giving 24% pass@1 with a 95% Wilson interval of 14.3%–37.4%. It recovered no frozen
successes.

The evidence supports four conclusions:

- The original 0/20 evaluation concealed learned probability mass.
- Repetition penalty 1.05 was harmful to this trained policy.
- The training decoder raises trained success further, from 24% to 32%, but it also
  changes temperature, probability truncation, and token budget. The additional eight
  percentage points cannot be attributed to one setting.
- Greedy output remains `click(13)`. Training improved the stochastic distribution but
  did not make the strict-valid action its mode.

## What the model learned

The defensible conclusion is:

> In this seed-23 run, GRPO increased strict-valid `click('13')` sampling on the fixed
> `click-test` prompt from 4% to 32% under its training decoder.

This is a positive single-task stochastic format-alignment result. It is not evidence
of:

- held-out task improvement;
- browser-control generalization;
- robust greedy behavior;
- reproducibility across training seeds; or
- an effect attributable to GRPO independently of the formatting curriculum.

The task contains one prompt, and the prompt includes the answer format. The result is
therefore deliberately narrow.

## Provenance and integrity

- Adapter SHA-256:
  `5b76ee537e2f39115090f7cf3afec6b6e766fe92a1376f80a2713e8de35f66ea`
- The hash matches `adapter_model.safetensors` in the original Phase 5 ZIP.
- Evaluator commit: `cf3dea74319dd506d25ac4845755948b6e95c360`
- GPU: Tesla T4, compute capability 7.5
- PyTorch: 2.11.0+cu128
- Transformers: 4.57.3
- TRL: 0.29.0
- PEFT: 0.19.1
- BrowserGym: 0.14.3
- Precision: FP32
- Peak allocated VRAM: 1,453,131,264 bytes, approximately 1.35 GiB
- vLLM and OpenEnv: not installed
- Stochastic generation seeds: 2000–2049, paired across frozen and trained policies

The raw manifest-bearing report is retained outside Git at
`artifacts/phase5/gate-a/phase5-gate-a-rescore.json`. The executed notebook is
`notebooks/phase5_gate_a_rescore.ipynb`.

## Decision

Cancel the proposed component-reward redesign. Its causal rationale is no longer
supported by the evidence.

The next sequence is:

1. Make the training decoder canonical for stochastic evaluation while reporting
   greedy output separately.
2. Remove repetition penalty from this task's evaluation path.
3. Add a decoder-parity test and ensure every executed run writes its manifest.
4. Verify TRL 0.29 `steps_per_generation` semantics with a production-facing contract
   test.
5. Configure one properly powered seed-23 run with deliberate group batching.
6. Run seeds 47 and 71 only after that configuration is frozen.

The central result is positive, but the stronger engineering story is the correction:
a mismatched deployment decoder hid learning that was present under stochastic
sampling.
