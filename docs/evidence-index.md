# Evidence index

Each claim, and the file in the [evidence bundle](https://huggingface.co/datasets/Aptheos/lfm-browser-control-evidence)
that supports it. `scripts/verify_reported_numbers.py` recounts the main ones from episodes, probes, tokens or update
records. `SHA256SUMS` covers every file except itself and the README, and the script pins its hash.

Commit hashes inside reports and in older result pages refer to the development history, not this branch. Most LFM3
reports also carry the SHA-256 of every source file they ran. The focused baseline and focused GRPO development reports and
the `release-check` reruns don't. The reruns used the clean copy of this repository.

| Claim | File | How it is derived |
| --- | --- | --- |
| Base gate 0/100 | `lfm3-book-flight/base-gate.json` | `episodes[].success` over seeds 10000–10099 |
| Reserved 160/200 | `lfm3-book-flight/reserved/report.json` | `episodes[].success` and `.reward` over seeds 10100–10299 |
| Reserved settings were locked | `lfm3-book-flight/reserved/frozen-protocol.json` | Task list and adapter hash; the report records the protocol's hash, though nothing public can prove when it was written |
| Failure types | `lfm3-book-flight/reserved/visible-audit.json` | Output of `scripts/audit_book_flight_traces.py` |
| 25/30 and 20/30 on development tasks | `lfm3-book-flight/primary-sft/development.json`, `focused-sft-baseline.json`, `primary-sft/comparison.json` | Seeds 70–99 |
| Replica 25/30 | `lfm3-book-flight/replica-sft/development.json` | Seeds 70–99 |
| SFT budget and settings | `lfm3-book-flight/sft-stage-1/`, `sft-stage-2/`, `primary-sft/`, `replica-sft/`: `training.json` | `config`, `updates`, `samples`, `inputs_sha256`, `changed_adapter_tensors` |
| Rejected pass 9/30 | `lfm3-book-flight/rejected-broader-pass/` | Episodes and paired rows |
| Failed first attempt at stage 3 | `lfm3-book-flight/stage-3-first-attempt/training.json` | Stopped after 21 updates |
| Default loss scale overflows | `lfm3-book-flight/inputs/gradient-check.json` | `attempts` |
| Base-model planning failure | `lfm3-book-flight/development-diagnostic.json` | `cells.C0`, `C1`, `C2` |
| First base-gate attempt | `lfm3-book-flight/base-gate-first-attempt.json` | 23 episodes before running out of memory |
| One-update trainer check | `lfm3-book-flight/one-update-contract/*.json` | Tensors changed, base hashes, gradient and mask checks |
| Probe and ID control scores | `lfm3-book-flight/*-sft/training.json` | `after`, `permuted_after`; `adoption.json` holds the decision |
| Visible decision audit | `lfm3-book-flight/*-sft/visible-audit.json` | Search fields and booking choice against the flights shown |
| Expert corpora and probes | `lfm3-book-flight/expert-corpus.json`, `inputs/` | Scripted solver; hashes match each `training.json` |
| Focused GRPO 21/30 | `lfm3-book-flight/focused-grpo/` | Seeds 70–99; `training.json` has the groups |
| Earlier variation | `lfm3-book-flight/earlier-variation/*.json` | Rewards within each task |
| Primary GRPO gate 0/6 | `lfm3-book-flight/primary-grpo-gate/rollouts.json` | `native_reward` and `completion_ids` within each task |
| `top_p` diagnostic 0/6 | `lfm3-book-flight/top-p-diagnostic/rollouts.json`, `control.json` | Same, with sampled log-probabilities; `decision.json` holds the stored reading |
| LFM3 click-test | `lfm3-click-test/frozen.json` | `conditions[].evaluation.episodes[].success` |
| Vision matrix | `vision/frozen-matrix/<cell>/report.json` | `conditions[].episodes[]`; screenshots beside each report |
| Vision variation | `vision/variation/top-p-095/report.json`, `top-p-100/report.json` | `groups[].episodes[]`, with raw log-probabilities |
| LFM2 click-test | `lfm2-click-test/seed-*-report.json` | `reward_records`, `frozen_evaluation`, `trained_evaluation` |
| Rerun of the published commands | `release-check/` | LFM3 25/30 for both adapters; LFM2 rescores and a seed-23 retrain |
| LFM2 decoder correction | `lfm2-click-test/gate-a-rescore.json` | Paired re-score: 2/50 frozen, 16/50 for the first checkpoint under the training decoder |

**Not recounted by the script:**

- Runtimes, and peak memory except the GRPO range.
- Reserved `top_k`, repetition penalty, thinking and attention settings. The report doesn't record the effective values; they come from the source, and `lfm3.py`'s hash in the report matches it.
- The base gate joining 35 saved and 65 resumed episodes, and the cause of the first attempt's failure, which is only in its log.
- SFT LoRA settings, loss scale, passes and replay, and the single zero-gradient update, which can all be read from `training.json`.
- Seeds 144–146 having varied before the focused GRPO run.
- The diagnostic's skill-level findings, click-test memory, and the figures on the July LFM2 pages other than the seed results.
- Rerun package versions and memory, which are in the rerun reports.

The script's checks can't show whether a report was honestly produced, only that the numbers follow from it.

The focused adapter that stage 3 starts from (`a920c16d…`) isn't published. Stages 1 and 2 can be retrained from the inputs and settings in the bundle, which won't give identical weights.
