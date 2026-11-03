# Browser control with small Liquid models

I train small [Liquid AI](https://huggingface.co/LiquidAI) models, [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B),
[LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) and, as a baseline so far, [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B),
to operate a real browser through [MiniWoB++](https://miniwob.farama.org/) tasks in
[BrowserGym](https://github.com/ServiceNow/BrowserGym), on a free T4 GPU. The policy reads the page and emits one tool call per turn, and the browser environment's
own reward decides success.

## How it works

Each turn the policy reads the page's accessibility tree and writes one tool call, such as `fill` a field or `click` an
element by its id. [BrowserGym](https://github.com/ServiceNow/BrowserGym) executes it in a real [Chromium](https://www.chromium.org/)
page, and [MiniWoB++](https://miniwob.farama.org/)'s own reward says whether the task
succeeded. No judge model is involved.

<p align="center">
  <img src="docs/media/book-flight-reserved-10100.gif" width="420"
       alt="LFM3-1.7B with the SFT adapter booking the shortest flight from HOT to PIE in ten steps">
</p>

Above, the SFT adapter books the shortest flight on [reserved task 10100](https://aptheos-browser-control-demo.static.hf.space/index.html#task=10100): two autocompletes, a date picker, a search,
then the shortest of three flights. Below, LFM2-350M on [`click-test`](https://miniwob.farama.org/environments/click-test/), before and after GRPO: the base model writes
text that isn't an action; the trained one writes `click('13')`.

| Base LFM2-350M | After GRPO |
| --- | --- |
| <img src="docs/media/click-test-base.gif" width="300" alt="Base model writes text instead of an action"> | <img src="docs/media/click-test-grpo.gif" width="300" alt="Trained model clicks the button"> |

These are replays, not screen recordings of a live run. MiniWoB tasks are seeded, so resetting a task and applying the
actions the model recorded during evaluation redraws exactly what it did; a replay that doesn't reach the recorded
reward fails. `scripts/render_episode_replay.py` makes them from the evaluation reports. You can step through more
episodes, including failures, and watch a narrated two-minute tour in the
[Space](https://huggingface.co/spaces/Aptheos/browser-control-demo).

## Results

| Model | Task | Method | Result |
| --- | --- | --- | --- |
| LFM3-1.7B | [`book-flight`](https://miniwob.farama.org/environments/book-flight/), multi-step, unseen tasks | Base model | 0/100 |
| LFM3-1.7B | [`book-flight`](https://miniwob.farama.org/environments/book-flight/), multi-step, unseen tasks | Supervised fine-tuning (SFT) on scripted expert traces | **160/200, 80.0%** (95% CI 73.9–85.0%) |
| LFM3-1.7B | [`book-flight`](https://miniwob.farama.org/environments/book-flight/), GRPO on top of SFT | Native reward | No detectable effect, and too little variation to continue |
| LFM3-VL-2B | [`click-color`](https://miniwob.farama.org/environments/click-color/), from screenshots | Base model, no training | 17/30 greedy, 15/30 sampled; 0/30 on a white image |
| LFM2-350M | [`click-test`](https://miniwob.farama.org/environments/click-test/), one fixed prompt | GRPO, 3 seeds | 2/50 → 50/50 for each seed |

All of it fits a free T4. The LFM3 reserved evaluation took about 1.6 hours for 200 episodes, including setup.

### LFM3-1.7B on [`book-flight`](https://miniwob.farama.org/environments/book-flight/)

The agent has to fill in origin and destination through autocomplete, work a read-only date picker, search, and book
the cheapest or shortest of the flights shown, on the `book-flight-nodelay` variant. The base model failed all 100
tasks, usually by reasoning past the token cap. When it did emit a call on separate development probes, it usually chose
the wrong step.

An SFT adapter trained on traces from a scripted solver then succeeded on 160 of 200 reserved tasks, which were
evaluated once, after the policy and decoder were locked. All 1,823 tool calls were well-formed, and 21 of them failed in
the browser, in 3 episodes. Most failures come at the final step: the agent reached the results page in 197 episodes and
booked correctly in 160. In 35 of the 40 failures it chose a flight visibly worse than the best one shown, and "shortest"
goals failed more often (28 of 105) than "cheapest" (12 of 95).

**What this does and doesn't show.** This is an estimate for one policy on one task family: simulated flights in
MiniWoB, under a sampling decoder at temperature 1 with `top_p` .95. The 0/100 and 160/200 used different seed ranges and
attention implementations, so they aren't a paired comparison. Policy selection used development tasks 70–99, which were reused across candidates, so the
development scores are not held-out estimates. Only the reserved number is.

### Why GRPO did nothing here

GRPO compares samples from the same task, and after the final SFT stage samples rarely differed. On 12 fresh tasks, four
samples each, every group matched token for token, so 0 of 12 groups varied in reward: 6 at temperature 1.25 with
`top_p` .95, and 6 at temperature 1 with no cutoff. At temperature 1, a whole episode reproduced with probability
.936–.987. Variation at that rate is rare rather than impossible, but it's too rare to fund GRPO at this budget. The two
failing tasks at temperature 1 were at .936 and .986, so failing didn't mean uncertain, though two tasks is a small sample.

An earlier 24-group run from a weaker SFT policy moved development tasks from 20/30 to 21/30, which is no detectable
effect. That policy did vary (2 of 8 groups at temperature 1, 3 of 8 at 1.25), and I picked 1.25 and some of the tasks
after seeing those checks, which favours finding an effect. This says GRPO had little signal on these policies, not that
GRPO can't help a less confident one. So there is no GRPO adapter from this work.

### LFM3-VL-2B baseline

Before any vision training I measured the base model on one-click tasks from screenshots only, with no accessibility
tree or element IDs. Pixels matter: on the two tasks that can succeed in one click, it scored 0 of 120 on a white image,
while most of those calls were well-formed. Failures on real screenshots are mostly malformed calls, which the strict
parser rejects without repair. A later variation check found real differences between samples of the untrained model,
though only 5 of 48 samples were valid clicks in the wrong place, so reward here would teach format as well as vision.
See the [vision baseline](docs/results/lfm3-vision-baseline.md). Vision training is the next stage of this project.

### LFM2-350M on [`click-test`](https://miniwob.farama.org/environments/click-test/)

This is the easy one, and its limit matters: `click-test` has a single fixed prompt, and a formatting curriculum was
active while the model first learned. It shows reproducible alignment to a strict action format, not generalization.
See the [three-seed report](docs/results/phase5-three-seed-reproducibility.md).

![Strict action success by GRPO group](docs/figures/three-seed-learning-trajectory.svg)

## Models and evidence

| Repository | Contents |
| --- | --- |
| [`Aptheos/lfm3-1.7b-book-flight-sft`](https://huggingface.co/Aptheos/lfm3-1.7b-book-flight-sft) | The 160/200 adapter, and a second-seed replica for corroboration |
| [`Aptheos/lfm2-350m-click-test-grpo`](https://huggingface.co/Aptheos/lfm2-350m-click-test-grpo) | Three GRPO adapters, seeds 23, 47 and 71 |
| [`Aptheos/lfm-browser-control-evidence`](https://huggingface.co/datasets/Aptheos/lfm-browser-control-evidence) | Raw reports with every episode, the expert corpus and screenshots |

The main results can be recounted offline from individual episodes, with no model or GPU. The script's bundle hash is
pinned, so an edit to any listed report or screenshot fails:

```bash
uvx --from huggingface_hub hf download Aptheos/lfm-browser-control-evidence --repo-type dataset --local-dir evidence
python scripts/verify_reported_numbers.py --evidence evidence
```

The [evidence index](docs/evidence-index.md) maps each claim to its file and lists what isn't recounted.

## Running it

Tests and lint need only the CPU and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.11
uv run pytest
uv run ruff check .
```

Without the GPU extras, 22 tests that need PyTorch, TRL, NumPy or Pillow are skipped.

[Reproducing the results](docs/reproducing.md) has the GPU commands, including a 30-task check of the published adapter.
On 2–3 November I reran the LFM3 adapter check and the LFM2 commands from a clean copy: both adapters scored 25/30 again, and LFM2 reached
50/50 again.

## Evaluation rules

- **Invalid output stays a failure.** A strict parser rejects malformed calls; nothing repairs them or retries.
- **Native reward only.** Success is the environment's terminal reward, with no shaping or oracle in evaluation.
- **Reserved tasks were used once.** The policy hash, decoder and task list were locked first, and no threshold or second attempt was involved.
- **GRPO masks are checked.** Browser observations are context and receive zero policy gradient, checked on the real model in a one-update run. See the [book-flight protocol](docs/lfm3-book-flight.md).
- **An element-ID control.** Consistently renumbering page IDs moved no skill by more than 1 of 20 probes, which is weak evidence against ID memorisation.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/browser_control/` | Environments, parsers, rollouts, evaluation and trainers |
| `scripts/` | Training, evaluation, contract checks, and the offline verifier |
| `tests/` | Unit, parser and trainer contract tests |
| `docs/` | Protocols and [result reports](docs/results/), including [LFM3 click-test](docs/results/lfm3-click-test.md) |
| `notebooks/` | Records of the July LFM2 runs on Colab |
| `docs/media/` | Episode replays rendered by `scripts/render_episode_replay.py` |

## License

The code is [MIT](LICENSE). The adapters are derived from Liquid AI's models and carry the `lfm1.0` licence, including
its commercial threshold; see each model card.
