"""Label click-color screenshots by trying the four box centres; only native reward picks the target."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing
from pathlib import Path

from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.vision_browser import CoordinateClick, VisionBrowserGymEnvironment

# click-color always draws a 2x2 grid; these are the box centres in normalized coordinates.
BOX_CENTRES = ((147, 481), (334, 481), (147, 781), (334, 781))


def label_task(environment, seed):
    """Return every centre the environment rewards, resetting before each attempt."""
    first = environment.reset(seed=seed)
    winners = []
    for x, y in BOX_CENTRES:
        environment.reset(seed=seed)
        step = environment.step(CoordinateClick(x, y))
        if step.reward > 0 and step.done:
            winners.append({"x": x, "y": y})
    return first.observation, winners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier evidence")
    if any(not 100 <= seed < 10000 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct training seeds 100..9999")
    images = args.output.parent / "expert-screenshots"
    images.mkdir(parents=True, exist_ok=True)
    root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
    examples, excluded = [], []
    with MiniwobAssetServer(root) as server:
        with closing(VisionBrowserGymEnvironment.miniwob("click-color", miniwob_url=server.base_url)) as env:
            for seed in args.seeds:
                observation, winners = label_task(env, seed)
                if len(winners) != 1:
                    excluded.append({"task_seed": seed, "winners": winners})
                    continue
                name = f"seed-{seed}.png"
                (images / name).write_bytes(observation.screenshot_png)
                examples.append(
                    {
                        "task_seed": seed,
                        "goal": observation.goal,
                        "width": observation.width,
                        "height": observation.height,
                        "screenshot": name,
                        "screenshot_sha256": hashlib.sha256(observation.screenshot_png).hexdigest(),
                        "target": winners[0],
                    }
                )
    args.output.write_text(json.dumps({"examples": examples, "excluded": excluded}, indent=2) + "\n")
    print({"examples": len(examples), "excluded": len(excluded)})
    if excluded:
        raise SystemExit("Some tasks had no single rewarded centre")


if __name__ == "__main__":
    main()
