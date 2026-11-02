# LFM3-1.7B on click-test

Frozen evaluation of `LiquidAI/LFM3-1.7B` at revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`, with no training. Three
overlapping RNG checks on a seed-invariant task, so these aren't independent replications and shouldn't be pooled.

| Interface | Greedy | Sampling, RNG 23 | RNG 47 | RNG 71 |
| --- | --- | --- | --- | --- |
| Native tool call | 3 of 3 | 50/50 | 50/50 | 50/50 |
| Strict text | 3 of 3 | 13/50 | 18/50 | 17/50 |

Every strict-text sampling failure was invalid formatting, often reasoning around the correct action. No completion
reached the 128-token cap. Peak allocated memory was about 3.24 GiB.

The two interfaces use different prompts, so this mixes prompting and interface effects. It supports skipping RL for the
native interface on this task, and nothing more: click-test has one fixed prompt and measures no generalization. The
project moved on to [book-flight](../lfm3-book-flight.md).
