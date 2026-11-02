# Phase 5 controlled pilot: seed 23

## Result

The controlled pilot completed on a Colab Tesla T4. Its original evaluation found
0/20 strict successes from both frozen and trained policies, but this is not a
held-out result: every `click-test` seed produces the same prompt, and the 20
episodes repeated one near-deterministic output. Evaluation also used a different
decoder from training. The result is therefore unresolved pending paired
re-evaluation of the existing checkpoint.

The final trained policy consistently emitted `click(13)`. That output is close
to the required `click('13')`, but the strict parser correctly rejects its
unquoted argument. The frozen policy instead emitted malformed prose/HTML.

## Run facts

- Model: `LiquidAI/LFM2-350M`
- Training seeds: 23–42
- Evaluation seeds: 1000–1019
- Execution: 60 optimizer steps over 15 four-completion generation groups
- Formatting shaping: linearly removed by optimizer step 40 (group 10)
- Precision: FP32
- Training loss: -0.056379
- Trainable LoRA tensors changed: 164/164
- Frozen strict success: 0/20; invalid action rate: 100%
- Trained strict success: 0/20; invalid action rate: 100%

Strict-valid `click('13')` actions appeared during training. There were no
successes among the first 36 completions, followed by 8/24 successes in the final
six groups; 7/20 occurred after shaping reached zero, and the final group reached
3/4. This strongly suggests format alignment but does not by itself establish a
controlled frozen-versus-trained improvement.

## Correction

My first reading of this run as negative was wrong, because its evaluation used a
different decoder from training. The paired re-evaluation in
[`phase5-gate-a-rescore.md`](phase5-gate-a-rescore.md) supersedes it.

The raw report is in the
[evidence bundle](https://huggingface.co/datasets/Aptheos/lfm-browser-control-evidence).
