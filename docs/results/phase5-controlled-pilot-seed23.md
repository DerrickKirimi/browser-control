# Phase 5 controlled pilot: seed 23

## Result

The controlled pilot completed on a Colab Tesla T4, but it did not improve the
held-out strict task metric. Both the frozen and trained policies scored 0/20 on
the fixed evaluation seeds 1000–1019, for a strict-success delta of zero.

The final trained policy consistently emitted `click(13)`. That output is close
to the required `click('13')`, but the strict parser correctly rejects its
unquoted argument. The frozen policy instead emitted malformed prose/HTML.

## Run facts

- Model: `LiquidAI/LFM2-350M`
- Training seeds: 23–42
- Evaluation seeds: 1000–1019
- Updates: 60; formatting shaping linearly removed by update 40
- Precision: FP32
- Training loss: -0.056379
- Trainable LoRA tensors changed: 164/164
- Frozen strict success: 0/20; invalid action rate: 100%
- Trained strict success: 0/20; invalid action rate: 100%

Strict-valid `click('13')` actions did appear in sampled training groups,
including after shaping reached zero. This shows that the policy can sample the
right grammar, but the final deterministic distribution mode did not retain the
quotes.

## Decision

Do not run seeds 47 and 71 with this unchanged curriculum. The generic string
similarity reward gives the nearly correct but invalid `click(13)` too much
credit. The next pilot should use explicit, tested format components—especially
quoted string arguments and exact lowercase action syntax—while preserving the
same strict held-out evaluation.

The raw report and adapter are retained locally under the ignored
`artifacts/phase5/seed-23/` directory.
