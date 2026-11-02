# LFM3-VL-2B vision baseline

Base `LiquidAI/LFM3-VL-2B` at revision `ba44bd51db20a2753f28c273a1571ccaa210d954`, with no training. The policy sees the
task instruction and a screenshot, never an accessibility tree or element IDs, and emits one native `click` with `x`
and `y` from 0 to 1000. See the [vision interface](../vision-grpo.md). These are development seeds, not held-out
estimates, and reserved vision seeds remain untouched.

## Frozen matrix

360 episodes on seeds 30–59, on a T4 in about 27 minutes. Peak allocated memory 3.6 GiB, and all logits finite.

| Native successes of 30 (valid calls) | click-color | click-shape | click-shades |
| --- | --- | --- | --- |
| Greedy, screenshot | 17 (17) | 5 (8) | 0 (0) |
| Sampling, screenshot | 15 (16) | 6 (8) | 0 (0) |
| Greedy, white image | 0 (27) | 0 (30) | 0 (30) |
| Sampling, white image | 0 (27) | 0 (30) | 0 (30) |

- **Pixels matter.** 0 of 120 on a white image for the two tasks that can succeed in one click, while most calls were valid.
- **Failures on real screenshots are mostly malformed calls.** In greedy `click-color`, all 17 valid calls succeeded and all 13 failures were malformed. On `click-shape` and `click-shades`, the model mostly writes a pair into one field, such as `<parameter=x>333, 755]`. The parser rejects these without repair.
- **`click-shades` looks impossible here.** Reading the task, its goal needs several selections and a submit, and this matrix allows one action. I didn't test that separately, but its 0/30 says little about vision either way.

## Within-group variation

Would GRPO have anything to compare? `click-color`, seeds 100–111, four samples per screenshot, temperature 1, top_p .95.
A group counts as a decision group if it holds both a success and a valid wrong click.

| 48 samples | Count |
| --- | --- |
| Success | 19 |
| Malformed | 24 |
| Valid, wrong click | 5 |

- 4 of 12 groups are decision groups, exactly the threshold of 4 I set in advance, though nothing public can prove that timing. 8 of 12 vary in native reward, and all 12 hold more than one token sequence.
- Whole-output probability has median .005 and maximum .116, against .94–.99 for the text SFT policy. This is the untrained base model, so it must be re-measured after any SFT.
- Only 5 samples carry the decision signal, and malformed output is the largest group. Native reward here teaches format as well as vision, so any fit must report valid-call rate and success among valid calls separately.
- Removing the `top_p` cutoff gave identical outcomes on all 48 samples. That arm shares random seeds with the first, so it isn't independent evidence, only a sign that the cutoff isn't what limits variation.
