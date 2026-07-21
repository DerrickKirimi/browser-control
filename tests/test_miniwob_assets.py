from pathlib import Path
from urllib.request import urlopen

import pytest

from browser_control.miniwob_assets import MiniwobAssetServer


def test_asset_server_exposes_miniwob_task(tmp_path: Path) -> None:
    task = tmp_path / "miniwob" / "click-test.html"
    task.parent.mkdir()
    task.write_text("<title>Click Test</title>", encoding="utf-8")

    with MiniwobAssetServer(tmp_path) as server:
        with urlopen(f"{server.base_url}click-test.html", timeout=2) as response:
            body = response.read().decode()

    assert response.status == 200
    assert body == "<title>Click Test</title>"


def test_asset_server_rejects_wrong_document_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid MiniWoB document root"):
        MiniwobAssetServer(tmp_path)
