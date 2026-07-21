"""Pinned MiniWoB assets and a local static server for Colab."""

from __future__ import annotations

import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

MINIWOB_REPOSITORY = "https://github.com/Farama-Foundation/miniwob-plusplus.git"
MINIWOB_REVISION = "eb59fed60fabe8951350275ba8650633b740013b"


def _git_output(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip()


def ensure_miniwob_assets(destination: str | Path) -> Path:
    """Fetch the pinned assets if absent and return their HTTP document root."""

    destination = Path(destination).expanduser().resolve()
    document_root = destination / "miniwob" / "html"
    expected_task = document_root / "miniwob" / "click-test.html"

    if destination.exists():
        if not expected_task.is_file():
            raise RuntimeError(
                f"MiniWoB destination exists but is incomplete: {destination}"
            )
        try:
            revision = _git_output("rev-parse", "HEAD", cwd=destination)
        except (FileNotFoundError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"MiniWoB destination is not a readable Git checkout: {destination}"
            ) from error
        if revision != MINIWOB_REVISION:
            raise RuntimeError(
                f"MiniWoB revision mismatch at {destination}: expected "
                f"{MINIWOB_REVISION}, found {revision}"
            )
        return document_root

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        _git_output(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            MINIWOB_REPOSITORY,
            str(destination),
        )
        _git_output("checkout", MINIWOB_REVISION, cwd=destination)
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"Failed to fetch MiniWoB assets into {destination}. Remove an incomplete "
            "destination before retrying."
        ) from error

    if not expected_task.is_file():
        raise RuntimeError(f"Pinned MiniWoB checkout lacks {expected_task}")
    return document_root


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args


class MiniwobAssetServer:
    """Serve MiniWoB files on an ephemeral localhost port."""

    def __init__(self, document_root: str | Path) -> None:
        root = Path(document_root).expanduser().resolve()
        if not (root / "miniwob" / "click-test.html").is_file():
            raise ValueError(f"Invalid MiniWoB document root: {root}")
        handler = partial(_QuietHandler, directory=str(root))
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}/miniwob/"

    def __enter__(self) -> MiniwobAssetServer:
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
