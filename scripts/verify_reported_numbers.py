"""Check the numbers reported in this repository against the raw run reports.

The reports are in the evidence bundle, not in Git. From the repository root:

    uvx --from huggingface_hub hf download Aptheos/lfm-browser-control-evidence \\
        --repo-type dataset --local-dir evidence
    python scripts/verify_reported_numbers.py --evidence evidence

This needs no model, GPU or third-party package. There are two kinds of check, counted separately:

- Integrity: the SHA-256 of the bundle's SHA256SUMS is pinned below, every listed file must match it, and no unlisted
  file may be present. This detects an edit to any listed report or screenshot, but not whether the original runs were
  honest, and anyone who edits this script as well can defeat it. The bundle's README isn't covered.
- Claims: the main values are recounted from episodes, probes, tokens or update records, then compared with the docs.

docs/evidence-index.md lists what isn't recounted. Pass --adapters DIR, holding the downloaded adapter repositories, to
also check the weight hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_control.book_flight_diagnostic import summarize_rollouts  # noqa: E402

BUNDLE_SHA256 = "9530b2ae78cda74463895e0bb048a40eb1f4143261ad1dc93cadd678eb715c7e"
Z = 1.959963984540054
PRIMARY_SFT = "53f80cc120b5b096f4c1d7d3d7490f13ff6c00e2f2877213b213e54e35791135"
REPLICA_SFT = "9840e091b5b9549d89de3a0281c7351c9d498aab0352639fbc316ac8855d0aa1"
FOCUSED_SFT = "a920c16d7f13de97167d55dc20416ca863f4cee6b983aae0a025158c0578b514"
LFM3_REVISION = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
GIB = 2**30
LFM2_SEEDS = {
    23: {"first_perfect": 8, "sha256": "268acb4c7459363d2d3f5812fb81fd797496c45c12f1b7104565434f3fc10ca8"},
    47: {"first_perfect": 6, "sha256": "05f3c54c3a4d8228df19e8c9651578b8bd5bc2deaa79a13b4bb973c17d9abf7e"},
    71: {"first_perfect": 3, "sha256": "aaf508b0784a79ecc7514096cafb098dec8cd9aba97ab51086f7ff41c6d39860"},
}
# Successes of 30, then valid calls, per vision cell and task.
VISION_MATRIX = {
    "greedy-screenshot": {"click-color": (17, 17), "click-shape": (5, 8), "click-shades": (0, 0)},
    "sampling-screenshot": {"click-color": (15, 16), "click-shape": (6, 8), "click-shades": (0, 0)},
    "greedy-white": {"click-color": (0, 27), "click-shape": (0, 30), "click-shades": (0, 30)},
    "sampling-white": {"click-color": (0, 27), "click-shape": (0, 30), "click-shades": (0, 30)},
}
BOOK = "lfm3-book-flight"
VISION_SFT = "1969938677a35eb4841677b0c2d200109903dc696f77c474626deebbfe324c1c"
VISION_GRPO = "98e13469e53c27cc1ee05f81ed905de087c7c685c06d5e214e5d565bc693d20d"
SEQUENCE_SFT = "df9c2e0918f6668bec9d7c3f5b2931240739e2a625542ce2183cd5535d0cddf6"
SEQUENCE_GRPO = "b0fe415c6b2e78b0c1052587d89b0a7d75fcc246e25157976bc35d0439ac138e"
GRPO_DEV = {"sampling-screenshot": (37, 39), "sampling-white": (0, 36), "greedy-screenshot": (38, 40), "greedy-white": (0, 40)}
SEQUENCE_RESERVED_B = {
    "sequence-sft": {"sampling-screenshot": (26, 30), "sampling-white": (0, 46), "greedy-screenshot": (91, 100)},
    "sequence-sft-grpo": {"sampling-screenshot": (83, 94), "sampling-white": (0, 83), "greedy-screenshot": (92, 100)},
}
# (successes, episodes where every call was valid) per policy and cell.
CLICK_COLOR_DEV = {
    "base": {"sampling-screenshot": (13, 14), "sampling-white": (0, 37), "greedy-screenshot": (21, 21), "greedy-white": (0, 37)},
    "sft": {"sampling-screenshot": (40, 40), "sampling-white": (0, 38), "greedy-screenshot": (40, 40), "greedy-white": (0, 40)},
    "grpo": {"sampling-screenshot": (40, 40), "sampling-white": (1, 40), "greedy-screenshot": (40, 40), "greedy-white": (6, 40)},
}
CLICK_COLOR_RESERVED = {
    "base": {"sampling-screenshot": (30, 31), "sampling-white": (0, 96), "greedy-screenshot": (63, 63)},
    "sft": {"sampling-screenshot": (100, 100), "sampling-white": (0, 99), "greedy-screenshot": (100, 100)},
    "grpo": {"sampling-screenshot": (100, 100), "sampling-white": (6, 98), "greedy-screenshot": (100, 100)},
}
SEQUENCE_DEV = {
    "baseline/base": {"sampling-screenshot": (0, 4), "sampling-white": (0, 12), "greedy-screenshot": (0, 5), "greedy-white": (0, 40)},
    "baseline/sft": {"sampling-screenshot": (1, 33), "sampling-white": (0, 27), "greedy-screenshot": (0, 40), "greedy-white": (0, 40)},
    "fit/sequence-sft": {"sampling-screenshot": (11, 12), "sampling-white": (0, 24), "greedy-screenshot": (37, 40), "greedy-white": (0, 40)},
}
SEQUENCE_RESERVED = {
    "base": {"sampling-screenshot": (0, 12), "sampling-white": (0, 26), "greedy-screenshot": (0, 13)},
    "sequence-sft": {"sampling-screenshot": (21, 22), "sampling-white": (0, 54), "greedy-screenshot": (95, 100)},
}


def wilson(successes: int, samples: int) -> tuple[float, float]:
    rate = successes / samples
    denominator = 1 + Z**2 / samples
    center = (rate + Z**2 / (2 * samples)) / denominator
    margin = Z * math.sqrt(rate * (1 - rate) / samples + Z**2 / (4 * samples**2)) / denominator
    return center - margin, center + margin


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Checks:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.claims = 0
        self.integrity = 0
        self.failures: list[str] = []

    def load(self, name: str) -> dict:
        return json.loads((self.root / name).read_text())

    def equal(self, label: str, actual: object, expected: object) -> None:
        if actual == expected:
            self.claims += 1
        else:
            self.failures.append(f"{label}: derived {actual!r}, reported {expected!r}")

    def hashed(self, label: str, actual: object, expected: object) -> None:
        if actual == expected:
            self.integrity += 1
        else:
            self.failures.append(f"{label}: {actual!r} differs from {expected!r}")


def wins(report: dict) -> int:
    return sum(episode["success"] for episode in report["episodes"])


def seeds(report: dict) -> list[int]:
    return [episode["task_seed"] for episode in report["episodes"]]


def evaluated(block: dict) -> int:
    return sum(episode["success"] for episode in block["evaluation"]["episodes"])


def check_integrity(c: Checks) -> None:
    sums = c.root / "SHA256SUMS"
    c.hashed("SHA256SUMS", sha256(sums), BUNDLE_SHA256)
    listed = set()
    for line in sums.read_text().splitlines():
        expected, name = line.split("  ", 1)
        listed.add(name)
        path = c.root / name
        c.hashed(f"file {name}", sha256(path) if path.is_file() else "missing", expected)
    present = {
        path.relative_to(c.root).as_posix()
        for path in c.root.rglob("*")
        if path.is_file() and path.relative_to(c.root).parts[0] != ".cache"
    }
    c.hashed("unlisted files", sorted(present - listed - {"SHA256SUMS", "README.md", ".gitattributes"}), [])


def check_lfm2(c: Checks, adapters: Path | None) -> None:
    for seed, published in LFM2_SEEDS.items():
        report = c.load(f"lfm2-click-test/seed-{seed}-report.json")
        records = report["reward_records"]
        groups = [records[i : i + 8] for i in range(0, len(records), 8)]
        counts = [sum(row["success"] for row in group) for group in groups]
        tag = f"LFM2 seed {seed}"
        c.equal(f"{tag} completions", len(records), 480)
        c.equal(f"{tag} first 8/8 group", counts.index(8), published["first_perfect"])
        strict = [count for count, group in zip(counts, groups) if group[0]["curriculum_weight"] == 0.0]
        c.equal(f"{tag} strict-reward successes", (len(strict), sum(strict)), (20, 160))
        for arm, sampled, greedy in (("frozen_evaluation", 2, 0), ("trained_evaluation", 50, 1)):
            block = report[arm]
            episodes = block["stochastic"]["evaluation"]["episodes"]
            c.equal(f"{tag} {arm} sampled", sum(e["success"] for e in episodes), sampled)
            c.equal(f"{tag} {arm} samples", len(episodes), 50)
            c.equal(f"{tag} {arm} greedy", sum(e["success"] for e in block["greedy"]["evaluation"]["episodes"]), greedy)
        if adapters:
            weights = adapters / "lfm2-350m-click-test-grpo" / f"seed-{seed}" / "adapter_model.safetensors"
            c.hashed(f"{tag} weights", sha256(weights), published["sha256"])
    rescored = c.load("lfm2-click-test/gate-a-rescore.json")
    c.equal("LFM2 first checkpoint, training decoder", evaluated(rescored["trained"]["training"]), 16)
    c.equal("LFM2 frozen, training decoder", evaluated(rescored["frozen"]["training"]), 2)


def check_lfm3_click_test(c: Checks) -> None:
    report = c.load("lfm3-click-test/frozen.json")
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    invalid_failures = failures = 0
    for condition in report["conditions"]:
        episodes = condition["evaluation"]["episodes"]
        cells[(condition["interface"], condition["decoder"]["name"])].append(sum(e["success"] for e in episodes))
        if condition["interface"] == "strict-text" and condition["decoder"]["name"] == "sampling":
            failures += sum(not e["success"] for e in episodes)
            invalid_failures += sum(not e["success"] and not e["valid_action"] for e in episodes)
    c.equal("LFM3 click-test native sampling", sorted(cells[("native-tools", "sampling")]), [50, 50, 50])
    c.equal("LFM3 click-test native greedy", cells[("native-tools", "greedy")], [1, 1, 1])
    c.equal("LFM3 click-test strict sampling", sorted(cells[("strict-text", "sampling")]), [13, 17, 18])
    c.equal("LFM3 click-test strict failures all invalid", (invalid_failures, failures), (102, 102))


def check_base_gate(c: Checks) -> None:
    base = c.load(f"{BOOK}/base-gate.json")
    episodes = base["episodes"]
    c.equal("base gate seeds", seeds(base), list(range(10000, 10100)))
    c.equal("base gate successes", wins(base), 0)
    c.equal("base gate stops", dict(Counter(e["stop_reason"] for e in episodes)), {"invalid-action": 98, "horizon": 2})
    capped = sum(e["stop_reason"] == "invalid-action" and e["steps"][-1]["completion_tokens"] == 128 for e in episodes)
    c.equal("base gate invalid outputs at the cap", capped, 97)
    c.equal("base gate upper Wilson", round(wilson(wins(base), len(episodes))[1] * 100, 1), 3.7)
    attempt = c.load(f"{BOOK}/base-gate-first-attempt.json")
    c.equal("first attempt was incomplete", (attempt["status"], len(attempt["episodes"])), ("running", 23))
    diagnostic = c.load(f"{BOOK}/development-diagnostic.json")["cells"]
    for cell, valid, correct in (("C0", 105, 24), ("C1", 102, 24), ("C2", 98, 32)):
        rows = [row for row in diagnostic[cell]["probes"] if row["skill"] != "navigate_calendar"]
        c.equal(f"diagnostic {cell} rollouts", sum(e["success"] for e in diagnostic[cell]["episodes"]), 0)
        c.equal(f"diagnostic {cell} probes", (len(rows), sum(r["valid"] for r in rows), sum(r["correct"] for r in rows)), (120, valid, correct))


def paired_totals(comparison: dict) -> tuple[int, int]:
    rows = comparison["paired"]
    return sum(r["baseline_success"] for r in rows), sum(r["candidate_success"] for r in rows)


def check_sft(c: Checks, adapters: Path | None) -> None:
    corpus_small = sha256(c.root / f"{BOOK}/inputs/expert-corpus-seeds100-131.json")
    corpus_full = sha256(c.root / f"{BOOK}/expert-corpus.json")
    gradient = c.load(f"{BOOK}/inputs/gradient-check.json")["attempts"][0]
    c.equal(
        "default loss scale overflow",
        (gradient["initial_scale"], gradient["nonfinite_gradient_tensors"], gradient["changed_tensors"]),
        (65536.0, 278, 0),
    )
    for name, updates, examples in (("sft-stage-1", 72, 288), ("sft-stage-2", 88, 352)):
        training = c.load(f"{BOOK}/{name}/training.json")
        sample_seeds = {row["task_seed"] for row in training["samples"]}
        c.equal(f"{name} budget", (len(training["updates"]), len(training["samples"])), (updates, examples))
        c.equal(f"{name} learning rate", training["config"]["learning_rate"], 1e-4)
        c.equal(f"{name} corpus", training["inputs_sha256"]["corpus"], corpus_small)
        c.equal(f"{name} seed range", (min(sample_seeds) >= 100, max(sample_seeds) <= 131), (True, True))
    stage_one = c.load(f"{BOOK}/sft-stage-1/training.json")
    c.equal("stage 1 rollouts", sum(e["success"] for e in stage_one["after_rollouts"]), 0)
    c.equal("stage 1 dates", sum(r["correct"] for r in stage_one["after"] if r["skill"] == "choose_date"), 0)

    focused = c.load(f"{BOOK}/focused-sft-baseline.json")
    c.equal("focused SFT development", wins(focused), 20)
    c.equal("focused SFT weights", focused["manifest"]["config"]["adapter_sha256"], FOCUSED_SFT)
    outcomes = {episode["task_seed"]: episode["success"] for episode in focused["episodes"]}
    for arm, weights, training_seed in (("primary-sft", PRIMARY_SFT, 23), ("replica-sft", REPLICA_SFT, 47)):
        path = f"{BOOK}/{arm}"
        development = c.load(f"{path}/development.json")
        training = c.load(f"{path}/training.json")
        comparison = c.load(f"{path}/comparison.json")
        decision = training["id_sensitivity_decision_all_skills"]["skills"]
        c.equal(f"{arm} development seeds", seeds(development), list(range(70, 100)))
        c.equal(f"{arm} development", wins(development), 25)
        c.equal(f"{arm} weights", development["manifest"]["config"]["adapter_sha256"], weights)
        c.equal(f"{arm} training seed", training["config"]["seed"], training_seed)
        c.equal(f"{arm} learning rate", training["config"]["learning_rate"], 3e-05)
        c.equal(f"{arm} initial weights", training["config"]["initial_adapter"]["weights_sha256"], FOCUSED_SFT)
        c.equal(f"{arm} budget", (len(training["updates"]), len(training["samples"])), (215, 860))
        c.equal(f"{arm} corpus", training["inputs_sha256"]["corpus"], corpus_full)
        c.equal(f"{arm} corpus seeds", len({row["task_seed"] for row in training["samples"]}), 94)
        c.equal(f"{arm} changed tensors", training["changed_adapter_tensors"], 284)
        c.equal(f"{arm} probes", sum(r["correct"] for r in training["after"] if r["skill"] != "navigate_calendar"), 118)
        drops = {
            skill: sum(r["correct"] for r in training["after"] if r["skill"] == skill)
            - sum(r["correct"] for r in training["permuted_after"] if r["skill"] == skill)
            for skill in decision
        }
        c.equal(f"{arm} ID drop at most 1 of 20", max(drops.values()) <= 1, True)
        c.equal(f"{arm} ID control recount", drops, {skill: row["drop"] for skill, row in decision.items()})
        won = sorted(e["task_seed"] for e in development["episodes"] if e["success"] and not outcomes[e["task_seed"]])
        lost = sorted(e["task_seed"] for e in development["episodes"] if outcomes[e["task_seed"]] and not e["success"])
        c.equal(f"{arm} paired wins", won, sorted(comparison["wins"]))
        c.equal(f"{arm} paired losses", lost, sorted(comparison["losses"]))
        c.equal(f"{arm} comparison rows", paired_totals(comparison), (20, 25))
    rejected = c.load(f"{BOOK}/rejected-broader-pass/comparison.json")
    c.equal("broader pass rejected", paired_totals(rejected), (20, 9))
    c.equal("broader pass episodes", wins(c.load(f"{BOOK}/rejected-broader-pass/development.json")), 9)
    if adapters:
        root = adapters / "lfm3-1.7b-book-flight-sft"
        c.hashed("primary weights file", sha256(root / "adapter_model.safetensors"), PRIMARY_SFT)
        c.hashed("replica weights file", sha256(root / "replica-seed-47" / "adapter_model.safetensors"), REPLICA_SFT)


def check_reserved(c: Checks) -> None:
    report = c.load(f"{BOOK}/reserved/report.json")
    protocol_path = c.root / f"{BOOK}/reserved/frozen-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    config = report["manifest"]["config"]
    episodes = report["episodes"]
    cohort = list(range(10100, 10300))
    c.equal("reserved seeds", seeds(report), cohort)
    c.equal("reserved protocol seeds", protocol["task_seeds"], cohort)
    c.equal("reserved protocol hash", config["frozen_protocol_sha256"], sha256(protocol_path))
    c.equal("reserved weights", config["adapter_sha256"], PRIMARY_SFT)
    c.equal("reserved revision", config["model_revision"], LFM3_REVISION)
    decoder = config["decoder"]
    c.equal(
        "reserved decoder",
        (decoder["do_sample"], decoder["temperature"], decoder["top_p"], decoder["max_new_tokens"]),
        (True, 1.0, 0.95, 128),
    )
    c.equal("reserved horizon", config["max_steps"], 15)
    count = sum(e["success"] for e in episodes)
    c.equal("reserved successes", count, 160)
    c.equal("reserved agrees with reward", sum(e["reward"] > 0 for e in episodes), count)
    low, high = wilson(count, len(episodes))
    c.equal("reserved Wilson interval", (round(low * 100, 1), round(high * 100, 1)), (73.9, 85.0))
    c.equal(
        "reserved stops",
        dict(Counter(e["stop_reason"] for e in episodes)),
        {"success": 160, "terminal-failure": 37, "horizon": 3},
    )
    errors = {e["task_seed"]: sum(step.get("action_error") is not None for step in e["steps"]) for e in episodes}
    c.equal("reserved action errors", {seed: n for seed, n in errors.items() if n}, {10162: 13, 10192: 6, 10199: 2})
    c.equal("reserved action-error episodes failed", any(e["success"] for e in episodes if errors[e["task_seed"]]), False)
    c.equal(
        "reserved generation seeds",
        all(step["generation_seed"] == e["task_seed"] * 16 + turn for e in episodes for turn, step in enumerate(e["steps"])),
        True,
    )
    recount = summarize_rollouts(episodes)
    c.equal(
        "reserved turns and validity",
        (recount["turns"], recount["valid_turns"], recount["first_turn_valid"]),
        (1823, 1823, 200),
    )
    calls = [step["text"] for episode in episodes for step in episode["steps"]]
    c.equal(
        "reserved raw text holds one tool call each",
        sum(text.count("<tool_call>") == 1 and text.count("</tool_call>") == 1 and text.count("<function=") == 1 for text in calls),
        1823,
    )
    reached = recount["milestones"]
    c.equal(
        "reserved milestones",
        (reached["origin_selected"], reached["destination_selected"], reached["date_correct"], reached["results_shown"]),
        (196, 198, 199, 197),
    )
    c.equal("reserved summary matches recount", report["summary"], recount)
    audit = c.load(f"{BOOK}/reserved/visible-audit.json")["episodes"]
    c.equal(
        "reserved visible audit",
        (
            sum(row["suboptimal_visible_booking"] and not row["native_success"] for row in audit),
            sum(row["suboptimal_visible_booking"] for row in audit),
            sum(row["search_with_incorrect_fields"] for row in audit),
        ),
        (35, 35, 4),
    )
    by_goal: Counter[str] = Counter()
    for episode in episodes:
        kind = "shortest" if "shortest" in episode["goal"] else "cheapest"
        by_goal[kind + ("-won" if episode["success"] else "-lost")] += 1
    c.equal(
        "reserved by goal type",
        dict(by_goal),
        {"shortest-won": 77, "shortest-lost": 28, "cheapest-won": 83, "cheapest-lost": 12},
    )
    corpus = (c.root / f"{BOOK}/expert-corpus.json").read_text()
    c.equal("reserved goals outside the corpus", sum(json.dumps(e["goal"])[1:-1] in corpus for e in episodes), 0)


def group_rollouts(episodes: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for episode in episodes:
        groups[episode["task_seed"]].append(episode)
    return groups


def varied_groups(groups: dict[int, list[dict]]) -> int:
    return sum(len({e["native_reward"] for e in group}) > 1 for group in groups.values())


def mask_checks(contract: dict) -> tuple[bool, bool]:
    return (
        all(row["external_gradients_zero"] for row in contract["loss_checks"]),
        all(row["raw_ids_preserved"] for row in contract["loss_checks"]),
    )


def check_grpo(c: Checks) -> None:
    for name, temperature, varied, tasks in (("t1", 1.0, 2, 8), ("t125", 1.25, 3, 8), ("t15", 1.5, 3, 4)):
        report = c.load(f"{BOOK}/earlier-variation/{name}.json")
        groups = group_rollouts(report["episodes"])
        c.equal(f"20/30 policy variation at {name}", (varied_groups(groups), len(groups)), (varied, tasks))
        c.equal(f"variation temperature {name}", report["manifest"]["config"]["decoder"]["temperature"], temperature)
    peaks = []
    for name in ("t1", "t125", "t15"):
        contract = c.load(f"{BOOK}/one-update-contract/{name}.json")
        c.equal(
            f"one-update contract {name}",
            (contract["changed_adapter_tensors"], contract["frozen_before_sha256"] == contract["frozen_after_sha256"]),
            (284, True),
        )
        c.equal(f"one-update masks {name}", mask_checks(contract), (True, True))
        peaks.append(contract["peak_allocated_bytes"])
    training = c.load(f"{BOOK}/focused-grpo/training.json")
    peaks.append(training["peak_allocated_bytes"])
    c.equal("GRPO peak memory range GiB", (round(min(peaks) / GIB, 1), round(max(peaks) / GIB, 1)), (4.9, 5.9))
    useful = sum(group["reward_variation"] for group in training["groups"])
    c.equal("focused GRPO budget", (training["optimizer_steps"], useful), (24, 5))
    grpo = c.load(f"{BOOK}/focused-grpo/development.json")
    comparison = c.load(f"{BOOK}/focused-grpo/comparison.json")
    c.equal("focused GRPO development", wins(grpo), 21)
    c.equal("focused GRPO comparison", paired_totals(comparison), (20, 21))

    for label, name, tasks, temperature, top_p, won in (
        ("gate", "primary-grpo-gate/rollouts.json", range(196, 202), 1.25, 0.95, 20),
        ("diagnostic", "top-p-diagnostic/rollouts.json", range(220, 226), 1.0, 1.0, 16),
    ):
        report = c.load(f"{BOOK}/{name}")
        decoder = report["manifest"]["config"]["decoder"]
        groups = group_rollouts(report["episodes"])
        c.equal(f"{label} settings", (decoder["temperature"], decoder["top_p"]), (temperature, top_p))
        c.equal(f"{label} tasks", sorted(groups), list(tasks))
        c.equal(f"{label} group sizes", {len(group) for group in groups.values()}, {4})
        c.equal(f"{label} varied groups", varied_groups(groups), 0)
        c.equal(f"{label} successes", sum(e["native_success"] for e in report["episodes"]), won)
        c.equal(
            f"{label} one token sequence per group",
            {len({json.dumps(e["completion_ids"]) for e in group}) for group in groups.values()},
            {1},
        )
    rollouts = c.load(f"{BOOK}/top-p-diagnostic/rollouts.json")["episodes"]
    logprobs = [[p for step in e["steps"] for p in step["sampling_logprobs"]] for e in rollouts]
    paths = [math.exp(sum(row)) for row in logprobs]
    tokens = [p for row in logprobs for p in row]
    c.equal("diagnostic path probability", (round(min(paths), 3), round(max(paths), 3)), (0.936, 0.987))
    c.equal("diagnostic lowest token probability", round(math.exp(min(tokens)), 3), 0.958)
    c.equal("diagnostic sampled tokens", len(tokens), 4092)
    failed = {e["task_seed"] for e in rollouts if not e["native_success"]}
    c.equal("diagnostic failing tasks", sorted(failed), [223, 225])
    control = group_rollouts(c.load(f"{BOOK}/top-p-diagnostic/control.json")["episodes"])
    c.equal("diagnostic control size", ({len(group) for group in control.values()}, len(control)), ({2}, 6))
    c.equal(
        "diagnostic matches control paths",
        all(rollout["completion_ids"] == control[rollout["task_seed"]][0]["completion_ids"] for rollout in rollouts),
        True,
    )


def check_release_rerun(c: Checks) -> None:
    """The 3 November rerun of the published commands and adapters, from a clean copy."""
    for name, arm in (("primary", "primary-sft"), ("replica", "replica-sft")):
        rerun = c.load(f"release-check/{name}.json")
        original = c.load(f"{BOOK}/{arm}/development.json")
        same = sum(a["success"] == b["success"] for a, b in zip(rerun["episodes"], original["episodes"]))
        text = sum(
            [s["text"] for s in a["steps"]] == [s["text"] for s in b["steps"]]
            for a, b in zip(rerun["episodes"], original["episodes"])
        )
        c.equal(f"rerun {name}", (wins(rerun), same, text), (25, 30, 29))
        c.equal(f"rerun {name} weights", rerun["manifest"]["config"]["adapter_sha256"], original["manifest"]["config"]["adapter_sha256"])
    for seed in LFM2_SEEDS:
        rescored = c.load(f"release-check/rescore-{seed}.json")
        c.equal(f"rerun LFM2 seed {seed}", (evaluated(rescored["trained"]["training"]), evaluated(rescored["frozen"]["training"])), (50, 2))
    retrain = c.load("release-check/lfm2-seed-23-report.json")
    c.equal(
        "rerun LFM2 training",
        (evaluated(retrain["frozen_evaluation"]["stochastic"]), evaluated(retrain["trained_evaluation"]["stochastic"]), evaluated(retrain["trained_evaluation"]["greedy"])),
        (2, 50, 1),
    )


def outcome(episode: dict) -> str:
    return "success" if episode["success"] else "wrong-click" if episode["valid_action"] else "malformed"


def check_screenshot(c: Checks, directory: Path, episode: dict) -> None:
    path = directory / episode["screenshot_filename"]
    c.hashed(f"screenshot {path.name}", sha256(path) if path.is_file() else "missing", episode["screenshot_sha256"])


def check_vision(c: Checks) -> None:
    for cell, tasks in VISION_MATRIX.items():
        folder = c.root / "vision/frozen-matrix" / cell
        report = json.loads((folder / "report.json").read_text())
        conditions = {condition["task"]: condition for condition in report["conditions"]}
        for task, (won, calls) in tasks.items():
            episodes = conditions[task]["episodes"]
            c.equal(f"vision {cell} {task} seeds", [e["task_seed"] for e in episodes], list(range(30, 60)))
            c.equal(f"vision {cell} {task} successes", sum(e["success"] for e in episodes), won)
            c.equal(f"vision {cell} {task} valid calls", sum(e["valid_action"] for e in episodes), calls)
            for episode in episodes:
                check_screenshot(c, folder / conditions[task]["screenshot_directory"], episode)
    arms = {}
    for name in ("top-p-095", "top-p-100"):
        folder = c.root / "vision/variation" / name
        arms[name] = json.loads((folder / "report.json").read_text())
        for group in arms[name]["groups"]:
            for episode in group["episodes"]:
                check_screenshot(c, folder / "screenshots", episode)
    report = arms["top-p-095"]
    outcomes: Counter[str] = Counter()
    decision_groups = native_groups = token_groups = 0
    paths = []
    for group in report["groups"]:
        kinds = [outcome(e) for e in group["episodes"]]
        outcomes.update(kinds)
        decision_groups += "success" in kinds and "wrong-click" in kinds
        native_groups += len({e["native_reward"] for e in group["episodes"]}) > 1
        token_groups += len({json.dumps(e["generation"]["completion_token_ids"]) for e in group["episodes"]}) > 1
        paths += [math.exp(sum(e["generation"]["raw_logprobs"])) for e in group["episodes"]]
    c.equal("vision variation outcomes", dict(outcomes), {"success": 19, "malformed": 24, "wrong-click": 5})
    c.equal("vision variation groups", (decision_groups, native_groups, token_groups), (4, 8, 12))
    c.equal("vision variation median path probability", round(statistics.median_high(paths), 3), 0.005)
    c.equal("vision variation maximum path probability", round(max(paths), 3), 0.116)
    second = [outcome(e) for group in arms["top-p-100"]["groups"] for e in group["episodes"]]
    first = [outcome(e) for group in report["groups"] for e in group["episodes"]]
    c.equal("vision second arm has identical outcomes", second, first)


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test on discordant pairs."""
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n) if n else 1.0


def vision_cell(c: Checks, path: Path, seeds: list[int], expected: tuple[int, int], label: str, adapter=None, protocol=None) -> dict:
    report = json.loads(path.read_text())
    c.equal(f"{label} complete", report["status"], "complete")
    episodes = report["conditions"][0]["episodes"]
    c.equal(f"{label} seeds", [e["task_seed"] for e in episodes], seeds)
    c.equal(f"{label} success is native terminal reward", all(e["success"] == (e["done"] and e["native_reward"] > 0) for e in episodes), True)
    c.equal(f"{label} successes and all-valid", (sum(e["success"] for e in episodes), sum(e["valid_action"] for e in episodes)), expected)
    c.equal(f"{label} adapter", report["manifest"]["config"]["adapter_sha256"], adapter)
    if protocol is not None:
        c.equal(f"{label} locked protocol", report.get("reserved_protocol_sha256"), protocol)
    return {e["task_seed"]: e["success"] for e in episodes}


def paired(a: dict, b: dict) -> tuple[int, int]:
    return sum(a[k] and not b[k] for k in a), sum(b[k] and not a[k] for k in a)


def check_vision_training(c: Checks, adapters: Path | None) -> None:
    root = c.root / "vision"
    fit = root / "click-color-fit"
    hashes = {"base": None, "sft": VISION_SFT, "grpo": VISION_GRPO}
    dev = {}
    for arm, cells in CLICK_COLOR_DEV.items():
        for cell, expected in cells.items():
            dev[arm, cell] = vision_cell(c, fit / f"eval/{arm}-{cell}/report.json", list(range(60, 100)), expected, f"click-color dev {arm} {cell}", hashes[arm])
    c.equal("click-color dev SFT vs base", paired(dev["sft", "sampling-screenshot"], dev["base", "sampling-screenshot"]), (27, 0))
    c.equal("click-color dev GRPO vs base", paired(dev["grpo", "sampling-screenshot"], dev["base", "sampling-screenshot"]), (27, 0))
    grpo = json.loads((fit / "grpo/training.json").read_text())
    varied = sum(check["varied"] for check in grpo["gradient_checks"])
    zero = all((check["norm"] > 0) == check["varied"] for check in grpo["gradient_checks"])
    c.equal("click-color GRPO varied groups, and zero gradient exactly when unvaried", (len(grpo["gradient_checks"]), varied, zero), (120, 18, True))
    reserved = root / "click-color-reserved"
    protocol = sha256(reserved / "reserved-protocol.json")
    res = {}
    for arm, cells in CLICK_COLOR_RESERVED.items():
        for cell, expected in cells.items():
            res[arm, cell] = vision_cell(c, reserved / f"reserved/{arm}-{cell}/report.json", list(range(21000, 21100)), expected, f"click-color reserved {arm} {cell}", hashes[arm], protocol)
    for arm in ("sft", "grpo"):
        wins, losses = paired(res[arm, "sampling-screenshot"], res["base", "sampling-screenshot"])
        c.equal(f"click-color reserved {arm} vs base", (wins, losses, sign_test(wins, losses) < 1e-20), (70, 0, True))
    sequence = {}
    for name, cells in SEQUENCE_DEV.items():
        stage, arm = name.split("/")
        folder = root / f"click-sequence-{stage}" / "eval"
        prefix = arm if stage == "baseline" else "sequence-sft"
        adapter = {"base": None, "sft": VISION_SFT, "sequence-sft": SEQUENCE_SFT}[arm]
        for cell, expected in cells.items():
            sequence[name, cell] = vision_cell(c, folder / f"{prefix}-{cell}/report.json", list(range(60, 100)), expected, f"click-sequence dev {name} {cell}", adapter)
    c.equal("click-sequence dev fit vs base", paired(sequence["fit/sequence-sft", "sampling-screenshot"], sequence["baseline/base", "sampling-screenshot"]), (11, 0))
    training = json.loads((root / "click-sequence-fit/sft/training.json").read_text())
    corpus = json.loads((root / "click-sequence-fit/experts/experts.json").read_text())
    c.equal("click-sequence SFT updates and examples", (len(training["updates"]), len(corpus["examples"])), (117, 234))
    c.equal("click-sequence excluded layouts", [e["task_seed"] for e in corpus["excluded"]], [140, 179, 223])
    c.equal("click-sequence SFT started from the click-color adapter", training["manifest"]["config"]["init_adapter_sha256"], VISION_SFT)
    variation = json.loads((root / "click-sequence-fit/variation-sequence-sft/report.json").read_text())["groups"]
    c.equal("click-sequence variation reward-varied groups", sum(g["summary"]["native_reward_varied"] for g in variation), 8)
    reserved = root / "click-sequence-reserved"
    protocol = sha256(reserved / "reserved-protocol.json")
    res = {}
    for arm, cells in SEQUENCE_RESERVED.items():
        for cell, expected in cells.items():
            res[arm, cell] = vision_cell(c, reserved / f"reserved/{arm}-{cell}/report.json", list(range(21000, 21100)), expected, f"click-sequence reserved {arm} {cell}", None if arm == "base" else SEQUENCE_SFT, protocol)
    wins, losses = paired(res["sequence-sft", "sampling-screenshot"], res["base", "sampling-screenshot"])
    c.equal("click-sequence reserved vs base", (wins, losses, sign_test(wins, losses)), (21, 0, 2 / 2**21))
    c.equal("click-sequence reserved greedy vs base", paired(res["sequence-sft", "greedy-screenshot"], res["base", "greedy-screenshot"]), (95, 0))
    grpo_root = root / "click-sequence-grpo"
    grpo = json.loads((grpo_root / "grpo/training.json").read_text())
    groups = grpo["groups"]
    zero_signal = all(not g["varied"] or g["nonzero_gradient"] for g in groups)
    c.equal("click-sequence GRPO groups, varied groups and updates", (len(groups), sum(g["varied"] for g in groups), grpo["optimizer_steps"]), (120, 39, 39))
    c.equal("click-sequence GRPO varied groups all had gradient", zero_signal, True)
    c.equal("click-sequence GRPO started from the SFT adapter and produced its hash", (grpo["manifest"]["config"]["init_adapter_sha256"], grpo["adapter_sha256"]), (SEQUENCE_SFT, SEQUENCE_GRPO))
    grpo_dev = {}
    for cell, expected in GRPO_DEV.items():
        grpo_dev[cell] = vision_cell(c, grpo_root / f"eval/sequence-grpo-{cell}/report.json", list(range(60, 100)), expected, f"click-sequence GRPO dev {cell}", SEQUENCE_GRPO)
    c.equal("click-sequence GRPO dev vs SFT, sampled", paired(grpo_dev["sampling-screenshot"], sequence["fit/sequence-sft", "sampling-screenshot"]), (26, 0))
    reserved_b = root / "click-sequence-reserved-b"
    protocol = sha256(reserved_b / "reserved-protocol.json")
    res_b = {}
    for arm, cells in SEQUENCE_RESERVED_B.items():
        for cell, expected in cells.items():
            res_b[arm, cell] = vision_cell(c, reserved_b / f"reserved/{arm}-{cell}/report.json", list(range(22000, 22100)), expected, f"click-sequence reserved B {arm} {cell}", SEQUENCE_GRPO if arm.endswith("grpo") else SEQUENCE_SFT, protocol)
    wins, losses = paired(res_b["sequence-sft-grpo", "sampling-screenshot"], res_b["sequence-sft", "sampling-screenshot"])
    c.equal("click-sequence reserved B GRPO vs SFT, sampled", (wins, losses, sign_test(wins, losses) < 1e-16), (57, 0, True))
    c.equal("click-sequence reserved B GRPO vs SFT, greedy", paired(res_b["sequence-sft-grpo", "greedy-screenshot"], res_b["sequence-sft", "greedy-screenshot"]), (1, 0))
    if adapters is not None:
        repo = adapters / "lfm3-vl-2b-click-sequence-sft"
        c.hashed("click-sequence GRPO adapter weights", sha256(repo / "sft-then-grpo/adapter_model.safetensors"), SEQUENCE_GRPO)
        c.hashed("click-sequence adapter weights", sha256(repo / "adapter_model.safetensors"), SEQUENCE_SFT)
        c.hashed("click-color adapter weights", sha256(repo / "click-color-sft/adapter_model.safetensors"), VISION_SFT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--adapters", type=Path)
    args = parser.parse_args()
    if not (args.evidence / "SHA256SUMS").is_file():
        print(f"No evidence bundle at {args.evidence}. See this script's docstring.")
        return 2
    c = Checks(args.evidence)
    check_integrity(c)
    for section in (
        lambda: check_lfm2(c, args.adapters),
        lambda: check_lfm3_click_test(c),
        lambda: check_base_gate(c),
        lambda: check_sft(c, args.adapters),
        lambda: check_reserved(c),
        lambda: check_grpo(c),
        lambda: check_vision(c),
        lambda: check_vision_training(c, args.adapters),
        lambda: check_release_rerun(c),
    ):
        try:
            section()
        except (OSError, KeyError, ValueError, TypeError) as error:
            c.failures.append(f"section stopped: {type(error).__name__}: {error}")
    print(f"{c.claims} claim checks and {c.integrity} integrity checks passed, {len(c.failures)} failed")
    for failure in c.failures:
        print("  FAIL", failure)
    return 1 if c.failures else 0


if __name__ == "__main__":
    sys.exit(main())
