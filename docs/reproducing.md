# Reproducing the results

These need a Linux machine with an NVIDIA T4 and Python 3.12. The evaluated settings assume a T4: vision refuses other
GPUs, and on newer GPUs the text path selects BF16 rather than the FP16 used here.

On Colab, make a separate Python 3.12 environment, since BrowserGym's pinned Playwright doesn't build on newer Python
versions, and installing into the notebook kernel won't work. On Ubuntu 24.04, which is what Colab runs, install these
first. I install them directly, and I haven't tested Playwright's `--with-deps` on this system:

```bash
apt-get update
apt-get install -y python3.12-venv libasound2t64 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 \
  libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libnss3 libnspr4 libpango-1.0-0 \
  libcairo2 libatspi2.0-0t64
```

I ran the LFM3 adapter check and all the LFM2 commands below from a clean copy of this repository, on a Colab T4, on
2–3 November 2026, with adapters copied in rather than downloaded. I tested the download separately. The results are under
`release-check/` in the evidence bundle, though those reports don't embed the commit or the LFM2 base revision, which was
`f37d3f5c8c5484bc01dad379a595cf4c68c4e70e`.

## LFM3-1.7B: check the published adapter (about 20 minutes)

This scores the adapter on development tasks 70–99, under the frozen settings. Reserved tasks are not used.

```bash
python3.12 -m venv .venv-lfm3
.venv-lfm3/bin/pip install -r requirements-lfm3.txt
.venv-lfm3/bin/pip install --no-deps -e .
.venv-lfm3/bin/python -m playwright install chromium
.venv-lfm3/bin/hf download Aptheos/lfm3-1.7b-book-flight-sft --include 'adapter_*' --local-dir adapters/lfm3-sft
.venv-lfm3/bin/python scripts/evaluate_book_flight_adapter.py --adapter adapters/lfm3-sft --output development.json
```

The recorded result is 25/30. The rerun also scored 25/30 for both the primary and the replica, with the same outcome on
all 30 tasks and identical text on 29. Runs on the same runtime can match this closely, but a different torch, driver or
GPU may not, so read a small difference as runtime noise rather than a failure.

## LFM3-1.7B: train

I haven't rerun training from a clean copy. Each stage's full configuration is in its `training.json` in the bundle, and
[the protocol](lfm3-book-flight.md) lists the stages. Each stage takes one T4 session, and `lfm3-book-flight/inputs/` in the
bundle holds the corpora, probes, validation reports and scripted records it used. Retraining won't give identical weights.

| Step | Script |
| --- | --- |
| Expert traces, in batches | `scripts/collect_book_flight_experts.py --start-seed 100 --episodes N --output FILE` |
| Merge batches | `scripts/merge_book_flight_experts.py --corpora FILE... --output FILE` |
| Check labels against the tokenizer | `scripts/check_book_flight_sft_tokens.py --corpus FILE --records RECORDS --output FILE` |
| Check for finite updates | `scripts/check_lfm3_gradients.py --records RECORDS --output FILE` |
| Train | `scripts/train_book_flight_sft.py --corpus FILE --gradient-report FILE --token-report FILE --probes FILE --output-dir DIR` |

`RECORDS` is the scripted solver's output, as in `inputs/scripted-records.json`. The trainer refuses to start without passing
reports matched to its model, corpus and encoder.

Stage 3 starts from stage 2's adapter through `--initial-adapter`, with a fresh optimizer. There's no resume: an
interrupted stage restarts. `scripts/build_book_flight_probes.py` documents how probes are built. Its `--validate` mode checks
them.

## LFM3-VL-2B: frozen matrix

The published runs used these pins, but I didn't rerun them from this clean copy.

```bash
python3.12 -m venv .venv-vision
.venv-vision/bin/pip install -r requirements-lfm3-vision.txt
.venv-vision/bin/pip install --no-deps -e .
.venv-vision/bin/python -m playwright install chromium
.venv-vision/bin/python scripts/evaluate_lfm3_vision.py --output frozen.json --decoder greedy --image-condition screenshot
.venv-vision/bin/python scripts/check_lfm3_vision_variation.py --output variation.json --task click-color \
  --seeds 100 101 102 103 104 105 106 107 108 109 110 111 --group-size 4 --top-p 0.95
```

Use `--decoder sampling` and `--image-condition white` for the other matrix cells.

## LFM2-350M: click-test GRPO (about 18 minutes per seed)

```bash
python3.12 -m venv .venv-lfm2
.venv-lfm2/bin/pip install -e ".[environment,model,training]" pytest
.venv-lfm2/bin/python -m playwright install chromium
.venv-lfm2/bin/python -m pytest -q tests/test_trl_contract.py tests/test_training.py tests/test_transformers_policy.py
.venv-lfm2/bin/python scripts/train_controlled.py --seed 23 --max-steps 60 --curriculum-steps 40 \
  --num-generations 8 --per-device-train-batch-size 8 --generation-batch-size 8 \
  --output-dir artifacts/phase5/full-group-seed-23
```

The July runs used PyTorch 2.11.0. This install leaves PyTorch unpinned, and for this rerun it took 2.15.0, with TRL
0.29.0, Transformers 4.57.3 and PEFT 0.19.1. That run again scored 2/50 frozen and 50/50 trained, with greedy success and
1.67 GiB peak allocated memory, against 1.73 GiB in July.

The final adapter is saved before evaluation, and there are no mid-run checkpoints: at this length, restarting a seed is
simpler and safer than an unverified GRPO resume. The notebooks in `notebooks/` are records of the July runs in the Colab
kernel of the time. Use the commands above today.
