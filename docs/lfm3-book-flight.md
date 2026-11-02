# LFM3-1.7B on book-flight: protocol and results

Model `LiquidAI/LFM3-1.7B` at revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`, on MiniWoB `book-flight-nodelay`
through BrowserGym 0.14.3. The main numbers can be recounted offline; see the [evidence index](evidence-index.md).

## Interface

Each turn the policy sees the conversation so far, including the accessibility tree, and emits exactly one native XML
call: `click`, `noop`, `fill` or `press`. Element IDs must pass a strict validator, fill values are bounded strings,
and keys come from a navigation allowlist. Fill triggers autocomplete but can't bypass read-only fields. No hidden state
or direct DOM writes are used.

Observations are context, never relabelled as policy output. Invalid XML ends the episode with zero reward. A valid
action that fails in the browser returns its error as an observation, so the policy can recover within the horizon.

## Frozen evaluation settings

| Setting | Value |
| --- | --- |
| Decoder | Sampling, temperature 1.0, top_p 0.95, top_k off, repetition penalty 1.0, thinking off |
| Completion cap | 128 tokens per call |
| Horizon | 15 model turns |
| Generation seed | `task_seed * 16 + turn` |
| Prompt budget | 16,384 tokens, no truncation |
| Success | BrowserGym terminal reward above 0 |
| Runtime | T4, FP16, SDPA, 512-token prefill chunks. The base-model gate used eager attention |

## Task seeds

| Seeds | Use |
| --- | --- |
| 0–99 | Development: probes, confirmation, matched comparison on 70–99 |
| 100–9999 | Training. Expert corpus within 100–195; GRPO gates for the final policy on 196–225. The earlier focused GRPO run used seeds 144–176, inside the corpus range |
| 10000–10099 | Base-model gate |
| 10100–10299 | Reserved; used once, for the final SFT policy |

## Base-model gate: 0/100

95% Wilson interval 0–3.7%, which sent the project to an SFT warm-start. There were 98 invalid outputs and two horizon
limits; 97 of the invalid outputs reached the token cap. The cap only cut off reasoning. A separate development
diagnostic on single decisions showed a planning failure alongside the format one: the model moved on without selecting
the airport suggestion and filled the read-only date field instead of opening the calendar. Thinking mode gave no rollout
success. Peak allocated memory
was about 4.2 GiB.

The first attempt at this gate ran out of memory on unchunked eager attention after 23 episodes. The replacement used
512-token prefill chunks, checked on development inputs, and no prompt was tuned on the gate's tasks. The final report
joins 35 saved episodes with 65 resumed ones. So 0/100 and 160/200 differ in attention implementation as well as seeds.

## SFT

The corpus comes from a scripted solver using only the goal and accessibility tree. The first two stages used seeds
100–131, and the third used the 94 usable seeds in 100–195. An episode with any action error is excluded. The encoder
masks the full prompt, and only the current call through EOS receives labels.

| Stage | Updates | Examples | LR | Start | Development reading |
| --- | --- | --- | --- | --- | --- |
| Full pass | 72 | 288 | 1e-4 | New LoRA (r 32, alpha 64) | 0/10 native; date selection 0/20 |
| Focused continuation, date and flight skills, 4 passes with replay | 88 | 352 | 1e-4 | Previous | 20/30 on tasks 70–99 |
| Lower-rate pass over the corpus | 215 | 860 | 3e-5 | Previous | **25/30** on tasks 70–99 |

All updates use AdamW with 4 examples per update, and initial loss scale 1. The default FP16 scale of 65536 overflowed
278 of 284 gradient tensors and changed no weights, so a finite-update check is required before SFT. A broader pass at
1e-4 fell from 20/30 to 9/30 and was rejected. The first attempt at the lower-rate stage stopped after 21 updates on a
zero-gradient guard. I changed the guard to count such steps instead, and reran; one of the 215 updates in the adopted run
has a zero gradient.

A second run differing only in training seed (47, against 23) also reached 25/30 from the same start. It corroborates
the training recipe. It wasn't scored on reserved tasks, and it shares the same 30 development tasks, so the two scores
must not be pooled.

Single-decision probes on development seeds scored 118/120 for each. Renumbering element IDs throughout a probe's history
moved no skill by more than 1 of 20. With 20 probes per skill, small effects would be invisible, so this is a sensitivity
check only.

## Reserved evaluation: 160/200

One locked policy (`53f80cc1…`), tasks 10100–10299, first and only use, with no threshold and no competing arm. 80.0%,
95% Wilson interval 73.9–85.0%.

- **Stops:** 160 success, 37 terminal failures, 3 horizon.
- **Validity:** 1,823 of 1,823 calls valid, and every first turn valid.
- **Action errors:** 21 in total, in 3 episodes, all failures.
- **Milestones of 200:** origin 196, destination 198, date 199, results shown 197, success 160.
- **Failures:** in 35 of 40, the booked flight was visibly worse than the best shown, from `scripts/audit_book_flight_traces.py`. Goals asking for the shortest flight failed 28 of 105 times, and the cheapest 12 of 95.

These tasks are now used. No later text policy can be evaluated on them and described as untouched.

## GRPO

| Run | Start | Setting | Result |
| --- | --- | --- | --- |
| Focused learning, 24 groups | 20/30 policy | 4 samples, T 1.25 | 21/30; only 5 groups were useful. No detectable effect |
| Gate for the final policy, tasks 196–201 | 25/30 policy | 4 samples, T 1.25, top_p .95 | 0/6 groups varied, so no training ran |
| Diagnostic, tasks 220–225 | 25/30 policy | 4 samples, T 1, top_p 1.0 | 0/6 varied; one token sequence per group |

An earlier attempt at the focused run stopped during training on a raw-history guard, with a different seed list. The
20/30 policy did vary: 2 of 8 groups at T 1, 3 of 8 at T 1.25 and 3 of 4 at T 1.5. I chose T 1.25 after those checks,
and the focused run replaced some tasks with seeds 144–146, already seen to vary. One of the 5 useful groups came from
them. This enrichment favours finding an effect, so the null still stands.

The trainer path itself works. One real TRL 1.16 update on the pretrained model changed all 284 LoRA tensors, left the
base hashes unchanged, matched padded IDs and masks to the raw collector rows, and gave external tokens zero policy
gradient. Peak allocated training memory across the GRPO runs was 4.9–5.9 GiB.

The diagnostic tested whether the `top_p` cutoff was hiding variation, and it wasn't. With the cutoff off, every sampled
token still had raw probability of at least .958, and a whole episode reproduced with probability .936–.987. All six
paths matched a same-task control with the cutoff on (2 samples per task). The two failing tasks, 223 and 225, were at
.936 and .986, which is a small sample for how confident failures are. At these probabilities, about 11% of four-sample
groups (5–23% by task) would hold more than one token sequence, and fewer would differ in reward. Variation is rare rather
than impossible, and too rare to fund GRPO at this budget.

Every sampled token on these six tasks had probability of at least .958, so a `top_p` .95 cutoff keeps only that token. On
these paths, then, the frozen sampling decoder acts like greedy. The reserved run recorded no log-probabilities, so I can't
say how far that extends to it.

## Limits

- One task family, with simulated flights. Nothing here measures other sites.
- One training lineage selected on reused development tasks.
- The reserved estimate holds for this decoder and horizon.
- Element-ID renumbering is a sensitivity check, not proof of general grounding.
