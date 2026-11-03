# Browser control with small Liquid models

I train small [Liquid AI](https://huggingface.co/LiquidAI) models, [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B),
[LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) and, as a baseline so far, [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B),
to operate a real browser through [MiniWoB++](https://miniwob.farama.org/) tasks in
[BrowserGym](https://github.com/ServiceNow/BrowserGym), on a free T4 GPU. The policy reads the page and emits one tool call per turn, and the browser environment's
own reward decides success.

## What the model sees and writes

[MiniWoB++](https://miniwob.farama.org/) is a set of more than a hundred small web pages, each with an instruction and its
own scorer: "click the button", "book the shortest flight", "click the cyan box". A task is generated from a seed, so
seed 10100 always produces the same airports, dates and flights, and the page itself reports success with a reward of
1. [BrowserGym](https://github.com/ServiceNow/BrowserGym) runs these pages in a real browser and turns each one into a
turn-by-turn exchange with a model.

The model never touches the browser directly. Each turn it receives the instruction and a description of the page, and
it must answer with exactly one action. A strict parser reads that action, and anything else, an explanation, two
actions, a malformed call, ends the episode with reward 0. Nothing is repaired or retried.

**What it sees.** For the text models, the page arrives as an accessibility tree: the page's elements, one per line, each
interactive one with an id. This is the whole page for [`click-test`](https://miniwob.farama.org/environments/click-test/):

```text
Goal: Click the button.
RootWebArea 'Click Test Task', focused
	[13] button 'Click Me!'
```

and this is the start of a [`book-flight`](https://miniwob.farama.org/environments/book-flight/) page:

```text
Goal: Book the shortest one-way flight from: HOT to: PIE on 10/24/2016.
	[17] heading 'Book Your One-Way Flight'
	[19] textbox 'From:'
	[21] textbox 'To:'
	StaticText 'Departure Date'
	[25] textbox ''
	[27] button 'Search'
```

The vision model gets none of that: only a screenshot and the instruction, and it answers with a click position from 0
to 1000 on each axis.

**What it writes.** On `click-test` the whole correct answer is `click('13')`. On `book-flight`, LFM3 answers in its own
tool-calling format, one call per turn, and needs about ten turns: type the origin, pick it from the autocomplete list,
do the same for the destination, open the date picker, page to the right month, pick the day, search, then compare the
flights and book the right one. The first turn of the episode above is:

```text
<tool_call>
<function=fill>
<parameter=bid>
19
</parameter>
<parameter=value>
HOT
</parameter>
</function>
</tool_call>
```

**Why `click-test` matters even though it's trivial.** The prompt describes the action format and even gives
`click('13')` as its example, so the task tests one thing: whether the model will answer in the required form at all.
Base LFM2-350M mostly didn't. It wrote near misses such as `click('Click Test Task')`, with the page's title where the id
belongs, or `click(<button id="13">Click Me!</button>)`, with the button's HTML inside the call, and succeeded 2 times in
50. After GRPO, trained only on that reward, it wrote `click('13')` 50 times in 50, for each of three training seeds.
That's the first hurdle for any small model acting through tools: no task can be solved before the output is something
the environment can execute. `book-flight` then asks for the same discipline across ten dependent steps, plus the
judgement to pick the right flight.

## How it works

Each turn the policy reads the page's accessibility tree and writes one tool call, such as `fill` a field or `click` an
element by its id. [BrowserGym](https://github.com/ServiceNow/BrowserGym) executes it in a real [Chromium](https://www.chromium.org/)
page, and [MiniWoB++](https://miniwob.farama.org/)'s own reward says whether the task
succeeded. No judge model is involved.

<p align="center">
  <img src="docs/media/book-flight-reserved-10100.gif" width="420"
       alt="LFM3-1.7B with the SFT adapter booking the shortest flight from HOT to PIE in ten steps">
</p>

Above, the SFT adapter books the shortest flight on [reserved task 10100](https://huggingface.co/spaces/Aptheos/browser-control-demo?task=10100): two autocompletes, a date picker, a search,
then the shortest of three flights. Below, LFM2-350M on [`click-test`](https://miniwob.farama.org/environments/click-test/), before and after GRPO: the base model writes
something close to an action that the parser can't execute; the trained one writes `click('13')`.

| Base LFM2-350M | After GRPO |
| --- | --- |
| <img src="docs/media/click-test-base.gif" width="300" alt="Base model writes a call the parser cannot execute"> | <img src="docs/media/click-test-grpo.gif" width="300" alt="Trained model clicks the button"> |

These are replays, not screen recordings of a live run. MiniWoB tasks are seeded, so resetting a task and applying the
actions the model recorded during evaluation redraws exactly what it did; a replay that doesn't reach the recorded
reward fails. `scripts/render_episode_replay.py` makes them from the evaluation reports. You can step through more
episodes, including failures, and watch a narrated two-minute tour in the
[Space](https://huggingface.co/spaces/Aptheos/browser-control-demo).

## Results

| Model | Task | Method | Result |
| --- | --- | --- | --- |
| LFM3-1.7B | [`book-flight`](https://miniwob.farama.org/environments/book-flight/), multi-step, unseen tasks | Base model, one call per turn, thinking off, 128 tokens | 0/100 |
| LFM3-1.7B | [`book-flight`](https://miniwob.farama.org/environments/book-flight/), multi-step, unseen tasks | Supervised fine-tuning (SFT) on scripted expert traces | **160/200, 80.0%** (95% CI 73.9–85.0%) |
| LFM3-1.7B | [`book-flight`](https://miniwob.farama.org/environments/book-flight/), GRPO on top of SFT | Native reward | No detectable effect, and too little variation to continue |
| LFM3-VL-2B | [`click-color`](https://miniwob.farama.org/environments/click-color/), from screenshots | Base model, no training | 17/30 greedy, 15/30 sampled; 0/30 on a white image |
| LFM2-350M | [`click-test`](https://miniwob.farama.org/environments/click-test/), one fixed prompt | GRPO, 3 seeds | 2/50 → 50/50 for each seed |

All of it fits a free T4. The LFM3 reserved evaluation took about 1.6 hours for 200 episodes, including setup.

### LFM3-1.7B on [`book-flight`](https://miniwob.farama.org/environments/book-flight/)

The agent has to fill in origin and destination through autocomplete, work a read-only date picker, search, and book
the cheapest or shortest of the flights shown, on the `book-flight-nodelay` variant. Under this harness the base model
succeeded on none of 100 tasks. The harness asks for exactly one tool call per turn, with thinking off, at most 128 new
tokens, and a strict parser. In 97 of the 100 episodes the last turn was prose reasoning that reached the 128-token cap
before any call. On separate development probes, which start the model at a known step, 105 of 120 calls were
well-formed but only 24 matched an action that reaches that step's outcome; with thinking on and 512 tokens per turn it
was 32 of 120. That's a result for an untuned general model in this one-call format, not a measure of what LFM3 can do
elsewhere. The counts are in the evidence dataset (`lfm3-book-flight/base-gate.json`,
`lfm3-book-flight/development-diagnostic.json`), and `scripts/verify_reported_numbers.py` recounts each of them.

An SFT adapter trained on traces from a scripted solver then succeeded on 160 of 200 reserved tasks, which were
evaluated once, after the policy and decoder were locked. All 1,823 tool calls were well-formed, and 21 of them failed in
the browser, in 3 episodes. Most failures come at the final step: the agent reached the results page in 197 episodes and
booked correctly in 160. In 35 of the 40 failures it chose a flight visibly worse than the best one shown, and "shortest"
goals failed more often (28 of 105) than "cheapest" (12 of 95).

<p align="center">
  <img src="docs/media/book-flight-failure-10108.png" width="320"
       alt="Results page for reserved task 10108: the first flight shown takes 7h 29m">
</p>

A typical failure, reserved task 10108: the shortest flight, 7h 29m, is the first one listed, and the agent booked the
10h 41m flight further down instead. The form was filled correctly; the comparison went wrong.

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

<p align="center">
  <img src="docs/media/vision-screenshot-vs-white.png" width="520"
       alt="The base model clicks the cyan box on the real screenshot and a corner on a white image">
</p>

Base LFM3-VL-2B, greedy, on `click-color` task 30: the same prompt, with and without the pixels.

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

**Pinned versions.** The ML stack is pinned to the versions the results were produced with: PyTorch, Transformers, TRL,
PEFT, Accelerate, Pillow, Playwright and BrowserGym. Some of those pins have published security advisories, mostly for
Pillow's image decoders and Transformers' loaders. That's acceptable for local research runs on trusted inputs (MiniWoB's
own pages and Liquid AI's checkpoints), but don't reuse these pins in a service that handles untrusted images or
model files. Supporting packages without a bearing on the numbers are kept current.

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
