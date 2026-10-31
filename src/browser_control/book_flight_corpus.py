"""Merge immutable expert collections without changing their training weights."""

from __future__ import annotations

from copy import deepcopy

from browser_control.book_flight_sft import TRAINING_SEEDS, expert_examples

POLICY = "exclude-entire-episode-on-any-action-error"


def merge_expert_corpora(corpora: list[dict], *, solver_sha256: str) -> dict:
    if not corpora:
        raise ValueError("At least one completed corpus is required")
    if not isinstance(corpora[0].get("packages"), dict) or not corpora[0]["packages"]:
        raise ValueError("Corpus package provenance must be a nonempty mapping")
    merged = {
        "status": "complete",
        "task": "book-flight-nodelay",
        "policy": POLICY,
        "solver_sha256": solver_sha256,
        "packages": deepcopy(corpora[0]["packages"]),
        "seeds": [],
        "records": [],
        "failures": [],
        "examples": [],
    }
    used = set()
    for corpus in corpora:
        for key in ("status", "task", "policy", "solver_sha256", "packages"):
            if corpus.get(key) != merged[key]:
                raise ValueError(f"Corpus provenance differs in {key}")
        seeds = corpus["seeds"]
        if any(type(seed) is not int or seed not in TRAINING_SEEDS for seed in seeds):
            raise ValueError("Only training seeds 100..9999 are allowed")
        if len(set(seeds)) != len(seeds) or used.intersection(seeds):
            raise ValueError("Overlapping seeds would duplicate training weights")
        observed = [row["task_seed"] for row in corpus["records"] + corpus["failures"]]
        if len(observed) != len(set(observed)) or set(observed) != set(seeds):
            raise ValueError("Records/failures must cover declared seeds exactly once")
        expected = [
            example
            for record in corpus["records"]
            for example in expert_examples(record)
        ]
        if expected != corpus["examples"]:
            raise ValueError("Examples differ from clean successful raw expert records")
        for record in corpus["records"]:
            if record.get("eligible_for_sft") != bool(expert_examples(record)):
                raise ValueError("Record eligibility differs from the clean-data rule")
        used.update(seeds)
        for key in ("seeds", "records", "failures", "examples"):
            merged[key].extend(deepcopy(corpus[key]))
    merged["eligible_episodes"] = sum(
        record["eligible_for_sft"] for record in merged["records"]
    )
    return merged
