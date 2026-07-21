"""Reproducibility metadata for an experiment run."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Configuration and runtime facts needed to interpret a result."""

    run_name: str
    seed: int
    config: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str | None = field(default_factory=_git_commit)
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    packages: dict[str, str] = field(
        default_factory=lambda: _package_versions(
            ["torch", "transformers", "trl", "openenv-core", "vllm"]
        )
    )

    def write(self, path: str | Path) -> Path:
        """Write stable, human-readable JSON without creating parent directories."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path
