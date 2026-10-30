"""Build a code-only snapshot for the frozen Colab evaluation; never launches GPU."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/lfm3/prepared"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        [
            *root.glob("src/browser_control/*.py"),
            root / "scripts/evaluate_lfm3.py",
            root / "scripts/evaluate_book_flight.py",
            root / "scripts/check_book_flight.py",
            root / "scripts/measure_book_flight_prompts.py",
            root / "scripts/check_lfm3_trl_contract.py",
            root / "scripts/check_lfm3_prefill.py",
            root / "scripts/check_lfm3_numerics.py",
            root / "scripts/diagnose_book_flight.py",
            root / "scripts/preflight_lfm3_attention.py",
            root / "scripts/check_lfm3_gradients.py",
            root / "scripts/collect_book_flight_experts.py",
            root / "scripts/train_book_flight_sft.py",
            root / "scripts/check_book_flight_sft_tokens.py",
            root / "scripts/check_book_flight_rollout_tokens.py",
            root / "scripts/check_lfm3_rollout_contract.py",
            root / "scripts/permute_book_flight_probes.py",
            root / "scripts/evaluate_book_flight_id_probes.py",
            root / "scripts/colab_lfm3_frozen.py",
            root / "requirements-lfm3.txt",
            root / "pyproject.toml",
            root / "README.md",
        ]
    )
    provenance = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip(),
        "files_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    manifest = output / "source-manifest.json"
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    archive = output / "lfm3-source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in paths:
            bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
        bundle.add(manifest, arcname="source-manifest.json")
    driver = output / "colab_lfm3_frozen.py"
    driver.write_bytes((root / "scripts/colab_lfm3_frozen.py").read_bytes())
    print(archive)
    print("SHA256:", hashlib.sha256(archive.read_bytes()).hexdigest())
    print(driver)


if __name__ == "__main__":
    main()
