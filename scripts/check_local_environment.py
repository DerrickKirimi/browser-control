"""Run a model-free click-test diagnostic against local BrowserGym."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from browser_control.local_browsergym import LocalBrowserGymEnvironment
from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets


def _button_id(accessibility_tree: str) -> str:
    for line in accessibility_tree.splitlines():
        if "button" not in line.lower() or "click me" not in line.lower():
            continue
        match = re.search(r"\[(?:bid=)?([^\]]+)\]", line)
        if match:
            return match.group(1).strip()
    raise RuntimeError("Could not locate the click-test button in the accessibility tree")


def _run(miniwob_url: str) -> None:
    environment = LocalBrowserGymEnvironment.miniwob(
        "click-test", miniwob_url=miniwob_url
    )
    report: dict[str, object] = {
        "backend": "local-browsergym",
        "task": "click-test",
    }
    try:
        started = time.monotonic()
        initial = environment.reset(seed=23)
        report["reset_seconds"] = round(time.monotonic() - started, 3)
        report["goal"] = initial.observation.goal
        report["accessibility_tree_prefix"] = (
            initial.observation.accessibility_tree[:500]
        )

        started = time.monotonic()
        noop = environment.step("noop()")
        report["noop_seconds"] = round(time.monotonic() - started, 3)
        report["noop_reward"] = noop.reward
        report["noop_done"] = noop.done

        tree = noop.observation.accessibility_tree
        action = f"click('{_button_id(tree)}')"
        started = time.monotonic()
        success = environment.step(action)
        report["click_seconds"] = round(time.monotonic() - started, 3)
        report["click_action"] = action
        report["click_reward"] = success.reward
        report["click_done"] = success.done
    finally:
        environment.close()

    if report["click_reward"] <= 0 or not report["click_done"]:
        raise RuntimeError(f"Scripted click did not solve click-test: {report}")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("artifacts/miniwob-plusplus"),
        help="Pinned MiniWoB Git checkout (downloaded if absent)",
    )
    args = parser.parse_args()
    document_root = ensure_miniwob_assets(args.assets_dir)
    with MiniwobAssetServer(document_root) as server:
        _run(server.base_url)


if __name__ == "__main__":
    main()
