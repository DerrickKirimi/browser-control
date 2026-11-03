# Browser control with small Liquid models

I train small [Liquid AI](https://huggingface.co/LiquidAI) models, [LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B),
[LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) and [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M), to operate a
real browser through [MiniWoB++](https://miniwob.farama.org/) tasks in [BrowserGym](https://github.com/ServiceNow/BrowserGym),
on the free T4 GPUs of [Google Colab](https://colab.research.google.com/) and [Kaggle](https://www.kaggle.com/). Each turn the model sees the page, as a screenshot or as text, and emits one action toward
completing the task. Success is decided by the reward that each MiniWoB++ task computes itself, not by me or by another
model.

- **From pixels alone**, [LFM3-VL-2B tuned with SFT, then GRPO](https://huggingface.co/Aptheos/lfm3-vl-2b-click-sequence-sft)
  clicks through a two-button sequence on **92 of 100** held-out tasks greedy and **83 of 100** sampled at temperature 1;
  the base model solved none in this experiment. GRPO alone took sampled success from 26 to 83.
- **From the page's text**, a [LoRA-tuned LFM3-1.7B](https://huggingface.co/Aptheos/lfm3-1.7b-book-flight-sft) books the right flight on **160 of 200** held-out ten-step tasks; the
  base model solved none in this experiment's one-call format.
- **With reinforcement learning only**, [LFM2-350M](https://huggingface.co/Aptheos/lfm2-350m-click-test-grpo) goes from 2/50 to 50/50 on its first browser task.

## How it works

<p align="center">
  <img src="docs/media/click-sequence-reserved-21001.gif" width="360"
       alt="LFM3-VL-2B with the sequence adapter clicking ONE then TWO on reserved task 21001, from the screenshot alone">
</p>

Above, the [vision adapter](https://huggingface.co/Aptheos/lfm3-vl-2b-click-sequence-sft) on [reserved task 21001](https://huggingface.co/spaces/Aptheos/browser-control-demo?task=21001) of
[`click-button-sequence`](https://miniwob.farama.org/environments/click-button-sequence/): it sees only the screenshot and
the instruction, and answers each turn with a click position. The red ring marks where its next click lands. The buttons
overlap on this task, and it clicks the visible part of ONE, then TWO.

The text models work the same way but read the page's accessibility tree instead of pixels, and write one tool call,
such as `fill` a field or `click` an element by its id. [BrowserGym](https://github.com/ServiceNow/BrowserGym) executes it in a real [Chromium](https://www.chromium.org/)
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

## What the model sees and writes

[MiniWoB++](https://miniwob.farama.org/) is a set of more than a hundred small web pages, each with an instruction and its
own scorer: "click the button", "book the shortest flight", "click the cyan box". A task is generated from a seed, so
seed 10100 always produces the same airports, dates and flights. When an episode ends, the task's scorer returns a reward:
1 for success, 0 for failure. [BrowserGym](https://github.com/ServiceNow/BrowserGym) runs these pages in a real browser
and turns each one into a turn-by-turn exchange with a model.

The model never touches the browser directly. Each turn it receives the instruction and a description of the page, and
it must answer with exactly one action in the required format. A strict parser reads that action; any other answer
ends the episode with reward 0, whether it's an explanation, two actions, or a call that's almost right. Nothing is
repaired or retried.

**What it sees.** For the text models, the page arrives as an accessibility tree: the page's elements, one per line, each
interactive one with an id. On the left is the page as a person sees it; on the right, the whole of what the text model
receives.

<table>
<tr><th><a href="https://miniwob.farama.org/environments/click-test/"><code>click-test</code></a>: the page</th><th>What the model receives</th></tr>
<tr><td><img src="docs/media/page-click-test.png" width="200" alt="The click-test page: one button labelled Click Me!"></td>
<td><pre>Goal: Click the button.
RootWebArea 'Click Test Task', focused
	[13] button 'Click Me!'</pre></td></tr>
<tr><th><a href="https://miniwob.farama.org/environments/book-flight/"><code>book-flight</code></a>: the page</th><th>What the model receives (first lines)</th></tr>
<tr><td><img src="docs/media/page-book-flight.png" width="200" alt="The book-flight form: From, To, Departure Date and Search"></td>
<td><pre>Goal: Book the shortest one-way flight from: HOT to: PIE on 10/24/2016.
	[17] heading 'Book Your One-Way Flight'
	[19] textbox 'From:'
	[21] textbox 'To:'
	StaticText 'Departure Date'
	[25] textbox ''
	[27] button 'Search'</pre></td></tr>
</table>

The vision model gets none of that: only the screenshot and the instruction. It answers with a click position from 0 to
1000 on each axis, measured across the whole screenshot, as below. Its first click on reserved task 21001 lands on the
visible part of ONE.

<p align="center">
  <img src="docs/media/vision-coordinates.png" width="520"
       alt="The click-button-sequence screenshot with a 0 to 1000 grid and the model's first click at x=307, y=757 on button ONE">
</p>

**What it writes.** On `click-test` the whole correct answer is `click('13')`. On `book-flight`, LFM3 answers in its own
tool-calling format, one call per turn, and needs about ten turns: type the origin, pick it from the autocomplete list,
do the same for the destination, open the date picker, page to the right month, pick the day, search, then compare the
flights and book the right one. The first turn of the episode above, and what it does to the page:

<table>
<tr><th>The model's first answer</th><th>The page afterwards</th></tr>
<tr><td><pre>&lt;tool_call&gt;
&lt;function=fill&gt;
&lt;parameter=bid&gt;
19
&lt;/parameter&gt;
&lt;parameter=value&gt;
HOT
&lt;/parameter&gt;
&lt;/function&gt;
&lt;/tool_call&gt;</pre>
It fills element 19, the <code>From:</code> box, with <code>HOT</code>; the page opens its autocomplete list, and the next
turn picks the airport from it.</td>
<td><img src="docs/media/book-flight-after-first-call.png" width="260" alt="The From box filled with HOT and its autocomplete list open"></td></tr>
</table>

**Why `click-test` matters even though it's trivial.** The prompt describes the action format and even gives
`click('13')` as its example, so the task tests one thing: whether the model will answer in the required form at all.
Base LFM2-350M mostly didn't. It wrote near misses such as `click('Click Test Task')`, with the page's title where the id
belongs, or `click(<button id="13">Click Me!</button>)`, with the button's HTML inside the call, and succeeded 2 times in
50. After GRPO, trained only on that reward, it wrote `click('13')` 50 times in 50, for each of three training seeds.
That's the first hurdle for any small model acting through tools: no task can be solved before the output is something
the environment can execute. `book-flight` then asks for the same discipline across ten dependent steps, plus the
judgement to pick the right flight.

## Results

| Model | Task | Method | Result |
| --- | --- | --- | --- |
| LFM3-VL-2B | [`click-button-sequence`](https://miniwob.farama.org/environments/click-button-sequence/), screenshots only, unseen tasks | Base model | 0/100 |
| LFM3-VL-2B | [`click-button-sequence`](https://miniwob.farama.org/environments/click-button-sequence/), screenshots only, unseen tasks | SFT on 234 expert clicks | **95/100 greedy** (95% CI 89–98%); 21/100 sampled at temperature 1 |
| LFM3-VL-2B | [`click-button-sequence`](https://miniwob.farama.org/environments/click-button-sequence/), fresh unseen tasks | SFT, then GRPO on native reward | **92/100 greedy**, **83/100 sampled** (SFT on the same tasks: 91 and 26) |
| LFM3-VL-2B | [`click-color`](https://miniwob.farama.org/environments/click-color/), screenshots only, unseen tasks | Base, then SFT | 30/100 → 100/100 sampled, mostly a format fix (see below) |
| LFM3-1.7B | [`book-flight`](https://miniwob.farama.org/environments/book-flight/), multi-step, unseen tasks | Base model, one call per turn, thinking off, 128 tokens | 0/100 |
| LFM3-1.7B | [`book-flight`](https://miniwob.farama.org/environments/book-flight/), multi-step, unseen tasks | Supervised fine-tuning (SFT) on scripted expert traces | **160/200, 80.0%** (95% CI 73.9–85.0%) |
| LFM3-1.7B | [`book-flight`](https://miniwob.farama.org/environments/book-flight/), GRPO on top of SFT | Native reward | No detectable effect, and too little variation to continue |
| LFM2-350M | [`click-test`](https://miniwob.farama.org/environments/click-test/), one fixed prompt | GRPO, 3 seeds | 2/50 → 50/50 for each seed |

All of it fits a free Colab or Kaggle T4. The LFM3 reserved evaluation took about 1.6 hours for 200 episodes, including setup. Every
"unseen tasks" number comes from a reserved seed range evaluated once, after the protocol, policy and decoder were locked.

### LFM3-VL-2B: clicking a sequence from pixels

The vision model gets no text description of the page at all, only the screenshot and the instruction "Click button ONE,
then click button TWO." It answers each turn with a click position from 0 to 1000 on each axis, sees the next screenshot,
and is told its earlier clicks. The two buttons appear anywhere on the page and often overlap.

In this experiment the base model solved none of 100 reserved tasks, mostly because its calls were malformed. I trained a [LoRA adapter](https://huggingface.co/Aptheos/lfm3-vl-2b-click-sequence-sft) on
234 expert turns from 117 training tasks: for each turn, the exact screenshot and prompt the model sees, and a click on
the visible part of the right button, kept only if the task's own scorer rewarded the episode. On 100 reserved tasks the adapter then
succeeded on **95 with greedy decoding** and **21 when sampling at temperature 1**. The locked test was the sampled one:
21 wins against the base model and no losses (exact sign test, p ≈ 1e-6). With a white image in place of the screenshot
it solved none, so the clicks depend on the pixels.

<p align="center">
  <img src="docs/media/compare-sequence-base-vs-sft-21001.gif" width="560"
       alt="Held-out task 21001, greedy: the base model's first answer is not a valid click; the SFT adapter clicks ONE then TWO">
</p>

**Then GRPO, to fix sampling.** Sampling was the weak spot: 78 of the 100 sampled episodes ended on a malformed call,
while episodes where every call was valid succeeded 95% of the time. The visual decision was learned; staying in format at
temperature 1 wasn't. Unlike the book-flight adapter, sampled episodes now varied in reward, which is what GRPO needs.

I used a simplified GRPO over whole episodes. For each of the same 120 training tasks it samples 4 episodes, scores each
only by the task's own reward, and weights every turn of an episode by that episode's reward minus its group's mean. That's
one on-policy update per group, with no clipping, KL penalty or standard-deviation scaling. Then a second locked
evaluation on 100 fresh tasks (seeds 22000–22099), against the SFT adapter:

| Policy | Sampled (locked test) | Every call valid, sampled | Greedy | White image |
| --- | --- | --- | --- | --- |
| SFT | 26/100 | 30 | 91/100 | 0/100 |
| **SFT, then GRPO** | **83/100** (95% CI 75–89%) | **94** | **92/100** | 0/100 |

GRPO won 57 tasks and lost none under sampling (exact sign test, p ≈ 1e-17), cost nothing on greedy, and still solves
nothing without the pixels. Success among fully valid episodes stayed at 87–88%, so what it learned is to stay in format.

<p align="center">
  <img src="docs/media/compare-sequence-sft-vs-grpo-22001.gif" width="560"
       alt="Held-out task 22001, sampled: the SFT adapter's answer is not a valid action; after GRPO it clicks ONE then TWO">
</p>

**What this does and doesn't show.** It's one MiniWoB task family with at most four clicks, not general computer use.
Both comparisons above are replays of the recorded held-out episodes, the same ones the numbers count.

**One click first.** Before the sequence, I trained a one-click adapter on
[`click-color`](https://miniwob.farama.org/environments/click-color/) ("click the cyan box"). On 100 reserved tasks it
went from 30 to 100 successes against the base model, but the base model already picked the right box in 30 of its 31
valid calls, so that gain is mainly the call format. A GRPO adapter matched it, but solved 6 of 100 tasks on white
images, above my declared limit of 5, so I don't claim it. The sequence adapter started from the one-click adapter, which
on its own solved 1 of 40 development sequences: it had the format, and it learned the sequence.

### LFM3-VL-2B before training

The untrained model already uses the pixels when it produces a valid call: on the one-click tasks it scored 0 of 120
on a white image. Its failures on real screenshots were mostly malformed calls, which the strict parser rejects. See the
[vision baseline](docs/results/lfm3-vision-baseline.md).

<p align="center">
  <img src="docs/media/vision-screenshot-vs-white.png" width="520"
       alt="The base model clicks the cyan box on the real screenshot and a corner on a white image">
</p>

Base LFM3-VL-2B, greedy, on `click-color` task 30: the same prompt, with and without the pixels.

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

### LFM2-350M on [`click-test`](https://miniwob.farama.org/environments/click-test/)

This is the easy one, and its limit matters: `click-test` has a single fixed prompt, and a formatting curriculum was
active while the model first learned. It shows reproducible alignment to a strict action format, not generalization.
See the [three-seed report](docs/results/phase5-three-seed-reproducibility.md).

![Strict action success by GRPO group](docs/figures/three-seed-learning-trajectory.svg)

## Models and evidence

| Repository | Contents |
| --- | --- |
| [`Aptheos/lfm3-vl-2b-click-sequence-sft`](https://huggingface.co/Aptheos/lfm3-vl-2b-click-sequence-sft) | The click-sequence SFT adapter, the SFT-then-GRPO adapter (`sft-then-grpo/`, 83/100 sampled), and the one-click `click-color` adapter they started from |
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

## Related work

Liquid AI's cookbook trains LFM2-350M on MiniWoB++ `click-test` with GRPO through TRL and OpenEnv
([example](https://github.com/Liquid4All/cookbook/tree/main/examples/browser-control),
[write-up by Pau Labarta Bajo](https://paulabartabajo.substack.com/p/fine-tuning-lfm2-350m-for-browser)), and names
`book-flight` and vision models as next steps. This project measures that `click-test` result across seeds and takes both
next steps, with held-out evaluation.

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
