"""Put replays of several policies on the same seeded task side by side, frame by frame.

Each input is a replay directory written by render_episode_replay.py for the same task seed. A shorter
episode holds its last frame while the others continue. Writes side-by-side.gif and its last frame.
Usage: --replays DIR [DIR ...] --labels LABEL [LABEL ...] --output DIR [--title TEXT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from browser_control.replay import side_by_side_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    if len(args.replays) != len(args.labels):
        parser.error("One label per replay")
    from PIL import Image

    seeds = {json.loads((d / "timeline.json").read_text())["task_seed"] for d in args.replays}
    if len(seeds) != 1:
        parser.error(f"Replays must share one task seed, got {sorted(seeds)}")
    columns = [[Image.open(p).convert("RGB") for p in sorted(d.glob("frame-*.png"))] for d in args.replays]
    outcomes = [json.loads((d / "timeline.json").read_text())["recorded_success"] for d in args.replays]
    frames = side_by_side_frames(columns, args.labels, outcomes, args.title)
    args.output.mkdir(parents=True, exist_ok=True)
    palette = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    durations = [1500] * (len(frames) - 1) + [3200]
    palette[0].save(args.output / "side-by-side.gif", save_all=True, append_images=palette[1:],
                    duration=durations, loop=0, optimize=True)
    frames[-1].save(args.output / "side-by-side-final.png")
    print(json.dumps({"seed": seeds.pop(), "frames": len(frames), "outcomes": dict(zip(args.labels, outcomes))}))


if __name__ == "__main__":
    main()
