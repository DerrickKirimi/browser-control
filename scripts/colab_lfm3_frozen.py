"""Colab-side driver: unpack a source snapshot, build the pinned runtime, and run one experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tarfile
from pathlib import Path


def sft_command(
    python: str,
    output: str,
    *,
    updates=16,
    passes=1,
    initial_adapter=None,
    focus_skills=None,
    id_permutation_seed=None,
    evaluate_permuted=False,
    replay_per_skill=0,
    learning_rate=1e-4,
    training_seed=23,
) -> list[str]:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("SFT learning rate must be finite and positive")
    if type(training_seed) is not int or not 0 <= training_seed < 2**32:
        raise ValueError("SFT training seed must be a nonnegative32-bit integer")
    command = [
        python,
        "scripts/train_book_flight_sft.py",
        "--corpus",
        "/content/lfm3-experts.json",
        "--gradient-report",
        "/content/lfm3-gradients.json",
        "--token-report",
        "/content/lfm3-sft-tokens.json",
        "--probes",
        "/content/lfm3-bookflight-probes.json",
        "--output-dir",
        output,
        "--updates",
        str(updates),
        "--passes",
        str(passes),
        "--learning-rate",
        str(learning_rate),
        "--training-seed",
        str(training_seed),
    ]
    if initial_adapter is not None:
        command += ["--initial-adapter", initial_adapter]
    if focus_skills:
        command += ["--focus-skills", *focus_skills, "--seen-training-fit"]
    command += ["--replay-per-skill", str(replay_per_skill)]
    if id_permutation_seed is not None:
        command += ["--id-permutation-seed", str(id_permutation_seed)]
    if id_permutation_seed is not None or evaluate_permuted:
        command += [
            "--permuted-probes",
            "/content/lfm3-bookflight-probes-permuted.json",
        ]
    return command


def main(
    experiment: str = "click-test",
    resume_from: str | None = None,
    *,
    sft_updates: int = 16,
    sft_learning_rate: float = 1e-4,
    sft_training_seed: int = 23,
    id_permutation_seed: int | None = None,
    initial_adapter: str | None = None,
    focus_skills: list[str] | None = None,
    passes: int = 1,
    evaluate_permuted: bool = False,
    replay_per_skill: int = 0,
    rollout_seeds: list[int] | None = None,
    rollout_group_size: int = 2,
    rollout_temperature: float = 1.0,
) -> None:
    if experiment not in {
        "click-test",
        "book-flight-pilot",
        "book-flight-heldout",
        "book-flight-diagnostic",
        "book-flight-gradients",
        "book-flight-sft-pilot",
        "book-flight-id-probes",
        "book-flight-token-rollouts",
        "book-flight-confirmation",
        "book-flight-runtime",
    }:
        raise ValueError("unknown frozen experiment")
    if (
        experiment in {"book-flight-token-rollouts", "book-flight-confirmation"}
        and initial_adapter is None
    ):
        raise ValueError("Frozen token rollouts require a saved adapter")
    run_directory_name = (
        "lfm3-frozen" if experiment == "click-test" else "lfm3-" + experiment
    )
    root = Path("/content") / run_directory_name
    # Refuse to mix artifacts from separate executions. Probe/download before retrying.
    root.mkdir(exist_ok=False)
    archive = Path("/content/lfm3-source.tar.gz")
    with tarfile.open(archive) as bundle:
        bundle.extractall(root, filter="data")
    source = json.loads((root / "source-manifest.json").read_text())
    for name, expected in source["files_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    env = dict(os.environ, MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false")
    log = root / "execution.log"

    def run(args: list[str]) -> None:
        # File output survives CLI disconnects; failures remain visible to the caller.
        with log.open("a") as stream:
            result = subprocess.run(
                args, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        print(log.read_text()[-6000:], flush=True)
        result.check_returncode()

    status = root / "status.json"
    status.write_text(json.dumps({"status": "running"}))
    try:
        run(["nvidia-smi"])
        # BrowserGym's pinned Playwright/greenlet stack needs Python 3.12.
        # Keep it separate from Colab's Python 3.13 notebook kernel.
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "python3.12-venv"])
        runtime = Path("/content/lfm3-runtime-py312")
        if not (runtime / "bin/python").exists():
            run(["/usr/bin/python3.12", "-m", "venv", str(runtime)])
        python = str(runtime / "bin/python")
        run([python, "-m", "pip", "install", "-r", "requirements-lfm3.txt"])
        run([python, "-m", "pip", "check"])
        run([python, "-m", "pip", "install", "--no-deps", "-e", "."])
        if experiment == "book-flight-id-probes":
            adapter = root / "adapter"
            with tarfile.open("/content/lfm3-adapter.tar.gz") as bundle:
                bundle.extractall(adapter, filter="data")
            run(
                [
                    python,
                    "scripts/evaluate_book_flight_id_probes.py",
                    "--probes",
                    "/content/lfm3-bookflight-probes.json",
                    "--permuted-probes",
                    "/content/lfm3-bookflight-probes-permuted.json",
                    "--adapter",
                    str(adapter),
                    "--training-corpus",
                    "/content/lfm3-experts.json",
                    "--output",
                    str(root / "id-probes.json"),
                ]
            )
            status.write_text(json.dumps({"status": "complete"}))
            return
        if experiment == "book-flight-gradients":
            run(
                [
                    python,
                    "scripts/check_lfm3_gradients.py",
                    "--records",
                    "/content/lfm3-bookflight-scripted.json",
                    "--output",
                    str(root / "gradients.json"),
                ]
            )
            status.write_text(json.dumps({"status": "complete"}))
            return
        # Playwright 1.44's --with-deps predates Ubuntu 24.04's t64 names.
        run(
            [
                "apt-get",
                "install",
                "-y",
                "-qq",
                "libasound2t64",
                "libatk1.0-0t64",
                "libatk-bridge2.0-0t64",
                "libcups2t64",
                "libdrm2",
                "libxkbcommon0",
                "libxcomposite1",
                "libxdamage1",
                "libxfixes3",
                "libxrandr2",
                "libgbm1",
                "libnss3",
                "libnspr4",
                "libpango-1.0-0",
                "libcairo2",
                "libatspi2.0-0t64",
            ]
        )
        run([python, "-m", "playwright", "install", "chromium"])
        if experiment == "book-flight-runtime":
            status.write_text(json.dumps({"status": "complete"}))
            return
        if experiment == "book-flight-confirmation":
            run(
                [
                    python,
                    "scripts/evaluate_book_flight_id_probes.py",
                    "--confirmation",
                    "--probes",
                    "/content/lfm3-bookflight-probes.json",
                    "--permuted-probes",
                    "/content/lfm3-bookflight-probes-permuted.json",
                    "--adapter",
                    initial_adapter,
                    "--output",
                    str(root / "confirmation.json"),
                ]
            )
            status.write_text(json.dumps({"status": "complete"}))
            return
        if experiment == "book-flight-token-rollouts":
            run(
                [
                    python,
                    "scripts/check_book_flight_token_rollouts.py",
                    "--adapter",
                    initial_adapter,
                    "--output",
                    str(root / "token-rollouts.json"),
                    "--group-size",
                    str(rollout_group_size),
                    "--temperature",
                    str(rollout_temperature),
                    *(["--seeds", *map(str, rollout_seeds)] if rollout_seeds else []),
                ]
            )
            status.write_text(json.dumps({"status": "complete"}))
            return
        if experiment == "book-flight-sft-pilot":
            run(
                sft_command(
                    python,
                    str(root / "sft"),
                    updates=sft_updates,
                    learning_rate=sft_learning_rate,
                    training_seed=sft_training_seed,
                    passes=passes,
                    initial_adapter=initial_adapter,
                    focus_skills=focus_skills,
                    id_permutation_seed=id_permutation_seed,
                    evaluate_permuted=evaluate_permuted,
                    replay_per_skill=replay_per_skill,
                )
            )
            status.write_text(json.dumps({"status": "complete"}))
            return
        if experiment == "book-flight-pilot":
            run(
                [
                    python,
                    "scripts/measure_book_flight_prompts.py",
                    "--records",
                    "/content/lfm3-bookflight-scripted.json",
                    "--output",
                    str(root / "prompt-lengths.json"),
                ]
            )
        if experiment == "book-flight-diagnostic":
            preflight = root / "preflight.json"
            if resume_from is None:
                run(
                    [
                        python,
                        "scripts/preflight_lfm3_attention.py",
                        "--records",
                        "/content/lfm3-bookflight-scripted.json",
                        "--output",
                        str(preflight),
                    ]
                )
                attention = json.loads(preflight.read_text())["selected_attention"]
            else:
                retained = json.loads(Path(resume_from).read_text())
                attention = retained["manifest"]["config"]["attention"]
            command = [
                python,
                "scripts/diagnose_book_flight.py",
                "--probes",
                "/content/lfm3-bookflight-probes.json",
                "--attention",
                attention,
                "--output",
                str(root / "diagnostic.json"),
            ]
            if resume_from is not None:
                command.extend(["--resume-from", resume_from])
            run(command)
            status.write_text(json.dumps({"status": "complete"}))
            return
        evaluation = (
            ["scripts/evaluate_lfm3.py", "--episodes", "50"]
            if experiment == "click-test"
            else [
                "scripts/evaluate_book_flight.py",
                "--split",
                experiment.removeprefix("book-flight-"),
            ]
        )
        if resume_from is not None:
            if experiment != "book-flight-heldout":
                raise ValueError(
                    "resume is supported only for the fixed book-flight gate"
                )
            evaluation.extend(["--resume-from", resume_from])
        run([python, *evaluation, "--output", str(root / "frozen.json")])
        status.write_text(json.dumps({"status": "complete"}))
    except BaseException as error:
        status.write_text(json.dumps({"status": "failed", "error": str(error)}))
        raise
    finally:
        evidence_path = (
            "/content/lfm3-evidence.tar.gz"
            if experiment == "click-test"
            else f"/content/{run_directory_name}-evidence.tar.gz"
        )
        with tarfile.open(evidence_path, "w:gz") as evidence:
            for name in (
                "frozen.json",
                "diagnostic.json",
                "preflight.json",
                "gradients.json",
                "id-probes.json",
                "sft/training.json",
                "status.json",
                "execution.log",
                "source-manifest.json",
                "prompt-lengths.json",
            ):
                path = root / name
                if path.exists():
                    evidence.add(path, arcname=name)
            training = root / "sft/training.json"
            if training.exists():
                checkpoint = json.loads(training.read_text()).get("last_checkpoint")
                if checkpoint:
                    evidence.add(root / "sft" / checkpoint, arcname="adapter")


if __name__ == "__main__":
    main()
