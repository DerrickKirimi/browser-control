"""Label click-button-sequence turns: click visible points of ONE then TWO, kept only if rewarded.

The page supplies the button positions for labelling only; each example holds the screenshot and the
exact prompt text the policy sees at that turn (goal plus earlier clicks), and the target click.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import closing
from pathlib import Path

from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.vision_browser import CoordinateClick, VisionBrowserGymEnvironment
from browser_control.vision_evaluation import history_goal

TASK = "click-button-sequence"
LABELS = ("ONE", "TWO")


VISIBLE_POINT = """([label, xs, ys]) => {
    const hits = [];
    for (const y of ys) for (const x of xs) {
        const element = document.elementFromPoint(x, y);
        const button = element && element.closest("button");
        if (button && button.textContent.trim() === label) hits.push([x, y]);
    }
    return hits;
}"""


def button_centre(page, label, grid=11):
    """The visible point of the labelled button nearest its centre, in 0..1000 viewport coordinates.

    Buttons can overlap, so the centre may belong to the other button; the browser's own hit test
    (elementFromPoint) decides which points really reach this one.
    """
    viewport = page.viewport_size
    width, height = viewport["width"], viewport["height"]
    buttons = page.get_by_role("button", name=label, exact=True)
    if buttons.count() != 1:
        raise ValueError(f"Expected one {label} button")
    box = buttons.bounding_box()
    xs = [box["x"] + box["width"] * (i + 0.5) / grid for i in range(grid)]
    ys = [box["y"] + box["height"] * (i + 0.5) / grid for i in range(grid)]
    hits = page.evaluate(VISIBLE_POINT, [label, xs, ys])
    if not hits:
        return None
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    x, y = min(hits, key=lambda point: (point[0] - cx) ** 2 + (point[1] - cy) ** 2)
    return round(x / width * 1000), round(y / height * 1000)


def label_task(environment, seed):
    step = environment.reset(seed=seed)
    page = environment._capture._environment.unwrapped.page
    goal, clicks, turns = step.observation.goal, [], []
    for label in LABELS:
        observation = step.observation
        point = button_centre(page, label)
        if point is None:  # fully covered by the other button: no click can reach it
            return turns, False
        x, y = point
        turns.append({"observation": observation, "prompt_text": history_goal(goal, clicks), "target": {"x": x, "y": y}})
        step = environment.step(CoordinateClick(x, y))
        clicks.append((x, y))
    return turns, bool(step.done and step.reward > 0)


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
        with closing(VisionBrowserGymEnvironment.miniwob(TASK, miniwob_url=server.base_url)) as env:
            for seed in args.seeds:
                turns, rewarded = label_task(env, seed)
                if not rewarded:
                    excluded.append({"task_seed": seed})
                    continue
                for number, turn in enumerate(turns):
                    observation = turn["observation"]
                    name = f"seed-{seed}-turn-{number}.png"
                    (images / name).write_bytes(observation.screenshot_png)
                    examples.append({
                        "example_id": f"{seed}-{number}",
                        "task_seed": seed,
                        "turn": number,
                        "goal": turn["prompt_text"],
                        "width": observation.width,
                        "height": observation.height,
                        "screenshot": name,
                        "screenshot_sha256": hashlib.sha256(observation.screenshot_png).hexdigest(),
                        "target": turn["target"],
                    })
    args.output.write_text(json.dumps({"task": TASK, "examples": examples, "excluded": excluded}, indent=2) + "\n")
    print({"tasks": len(args.seeds) - len(excluded), "examples": len(examples), "excluded": len(excluded)})
    # Some layouts cover ONE completely with TWO, so no click can solve them; a few are expected and
    # recorded. Many would mean the labelling itself is wrong.
    if len(excluded) > len(args.seeds) // 10:
        raise SystemExit("More than 10% of tasks were not rewarded by their expert clicks")


if __name__ == "__main__":
    main()
