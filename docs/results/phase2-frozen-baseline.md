# Phase 2 frozen-policy baseline

Run date: 2026-07-21

Source commit evaluated: `538bca494d678ba377c4440a13306b53d8e4e15d`
(`add github auth`: the frozen-policy evaluation pipeline plus secure Colab repository
authentication used for this execution)

## Runtime

- Platform: Google Colab
- GPU: NVIDIA Tesla T4
- Compute capability: 7.5
- GPU memory reported: 15,637,086,208 bytes (14.56 GiB)
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN: 91900
- Model: `LiquidAI/LFM2-350M`
- Model dtype: FP16
- Peak allocated VRAM: 729,918,976 bytes (0.68 GiB)

## Environment validation

The in-process BrowserGym diagnostic succeeded before model evaluation:

- reset: 1.752 seconds;
- `noop()`: reward 0.0, non-terminal, 1.674 seconds;
- `click('13')`: reward 1.0, terminal, 0.815 seconds.

This independently validates the environment and reward path on Colab.

## Frozen-policy result

- Episodes: 20, seeds 23–42
- Success rate: 0/20 (0%)
- Valid-action rate: 0/20 (0%)
- Mean generation time: 0.428 seconds
- Warm mean generation time excluding the first generation: 0.327 seconds
- Prompt length: 137 tokens in every episode
- Completion length: 15 tokens in every episode

All seeds produced the identical output:

```text
click(<button id="13">Click Me!</button>)
```

The output identifies the correct element and desired operation but violates the
BrowserGym action contract `click('13')`. Because strict parsing rejected it, no
model-generated action was sent to the environment. This is the intended baseline:
it separates policy-format failure from environment or reward failure.

The raw `frozen-policy.json` remains an uncommitted experiment artifact. The Colab
executed frozen-policy notebook committed in
`e190c3443c467d1436b49866abc89a567774d6bd` preserves the
full execution output.
