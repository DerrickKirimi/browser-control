"""Replay recorded MiniWoB episodes into annotated frames and a GIF, with no model.

MiniWoB is seeded and each report step stores the exact BrowserGym action, so
reset(seed) plus the recorded actions redraws what the policy did. The replay
must reach the recorded reward, or it is reported as a mismatch.
"""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path

from browser_control.miniwob_assets import MiniwobAssetServer, ensure_miniwob_assets
from browser_control.replay import annotate, describe_call, element_labels
from browser_control.vision_browser import _ScreenshotCapture


def sharp(env, scale: float) -> bytes:
    """Recapture the current page at `scale` times the pixel density, with BrowserGym's own capture."""
    import io

    from browsergym.core.observation import extract_screenshot
    from PIL import Image

    if scale <= 1:
        return env.image_png
    page = env._environment.unwrapped.page
    before = page.evaluate("[window.innerWidth, window.innerHeight, window.devicePixelRatio, scrollX, scrollY]")
    page._bgym_scale_factor = scale
    try:
        pixels = extract_screenshot(page)
    finally:
        page._bgym_scale_factor = 1.0
        # BrowserGym restores metrics with an explicit override; clear it so the page is exactly as before,
        # which coordinate clicks depend on.
        session = page.context.new_cdp_session(page)
        session.send("Emulation.clearDeviceMetricsOverride")
        session.detach()
    if page.evaluate("[window.innerWidth, window.innerHeight, window.devicePixelRatio, scrollX, scrollY]") != before:
        raise RuntimeError("Sharp capture changed the page's metrics")
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def replay(env, episode: dict, directory: Path, upscale: int = 1, scale: float = 1) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    seed, goal = episode["task_seed"], episode["goal"]
    env.reset(seed=seed)
    frames = [annotate(sharp(env, scale), goal, "Start: the page the model sees", f"seed {seed}", upscale)]
    timeline, reward, done = [], 0.0, False
    for number, step in enumerate(episode["steps"], start=1):
        labels = element_labels(step["observation"].get("accessibility_tree", ""))
        described = describe_call(step.get("text", ""), labels)
        # The verdict comes first, so a long invalid answer can't push it out of the caption.
        caption = f"Step {number}: {described}" if step.get("valid_action") else f"Step {number}: not a valid action; {described}"
        if step.get("valid_action") and step.get("action") and not done:
            result = env.step(step["action"])
            reward, done = result.reward, result.done
        status = f"seed {seed}  ·  reward {reward:g}" + ("  ·  done" if done else "")
        frames.append(annotate(sharp(env, scale), goal, caption, status, upscale))
        timeline.append({"step": number, "model_output": step.get("text"), "action": step.get("action"),
                         "caption": caption, "replayed_reward": reward, "recorded_reward": step.get("reward")})
    for index, frame in enumerate(frames):
        frame.save(directory / f"frame-{index:02d}.png")
    durations = [1600] + [1400] * (len(frames) - 2) + [3000]
    # One palette per frame without dithering: flat UI colours stay clean instead of grainy.
    from PIL import Image

    palette = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    palette[0].save(directory / "episode.gif", save_all=True, append_images=palette[1:],
                    duration=durations[: len(frames)], loop=0, optimize=True)
    summary = {"task_seed": seed, "goal": goal, "recorded_success": bool(episode.get("success")),
               "recorded_reward": episode.get("reward"), "replayed_reward": reward,
               "matches": (reward > 0) == bool(episode.get("success")), "steps": timeline}
    (directory / "timeline.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def marked(png: bytes, action: str | None, scale: float, factor: float = 1.0) -> bytes:
    """Draw a ring where a recorded mouse_click(x, y) lands.

    Actions are in screenshot pixels; `factor` converts them to the capture's pixels.
    """
    import io
    import re

    from PIL import Image, ImageDraw

    match = re.fullmatch(r"mouse_click\(([\d.]+), ([\d.]+)\)", action or "")
    if match is None:
        return png
    image = Image.open(io.BytesIO(png)).convert("RGB")
    x, y = float(match[1]) * factor, float(match[2]) * factor
    r, w = 7 * max(scale, 1), max(2, round(1.5 * max(scale, 1)))
    draw = ImageDraw.Draw(image)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(230, 30, 30), width=w)
    draw.line((x - 2 * r, y, x + 2 * r, y), fill=(230, 30, 30), width=max(1, w // 2))
    draw.line((x, y - 2 * r, x, y + 2 * r), fill=(230, 30, 30), width=max(1, w // 2))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def replay_vision(env, episode: dict, directory: Path, scale: float) -> dict:
    """Replay a screenshot policy's recorded clicks; each frame shows where the next click lands.

    Capturing can disturb a page that is then clicked by coordinates, so the outcome is checked on a
    replay with no capture, and every frame comes from a fresh reset plus the clicks before it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    seed, goal = episode["task_seed"], episode["goal"]
    turns = episode.get("turns") or [episode]
    actions = [t.get("action") if t.get("valid_action") else None for t in turns]

    def play(count):
        env.reset(seed=seed)
        reward, done = 0.0, False
        for action in actions[:count]:
            if action is None or done:
                break
            result = env.step(action)
            reward, done = result.reward, result.done
        return reward, done

    reward, done = play(len(actions))
    # Recorded clicks are in the policy's screenshot pixels; frames are captured at `scale` x CSS pixels.
    viewport = env._environment.unwrapped.page.viewport_size
    factor = (scale if scale > 1 else 1) * viewport["width"] / env.image_size[0]
    frames = []
    for number, (turn, action) in enumerate(zip(turns, actions), start=1):
        play(number - 1)
        coords = turn.get("normalized_coordinates")
        caption = (f"Click {number}: x={coords['x']}, y={coords['y']} (of 1000)" if action
                   else f"Click {number}: not a valid action")
        frames.append(annotate(marked(sharp(env, scale), action, scale, factor), goal, caption,
                               f"seed {seed}  ·  screenshot only", 1))
        if action is None:
            break
    play(len(actions))
    outcome = "success" if done and reward > 0 else "failure"
    frames.append(annotate(sharp(env, scale), goal, f"Result: {outcome}", f"seed {seed}  ·  reward {reward:g}", 1))
    for index, frame in enumerate(frames):
        frame.save(directory / f"frame-{index:02d}.png")
    from PIL import Image

    durations = [1500] * (len(frames) - 1) + [2600]
    palette = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    palette[0].save(directory / "episode.gif", save_all=True, append_images=palette[1:],
                    duration=durations, loop=0, optimize=True)
    summary = {"task_seed": seed, "goal": goal, "recorded_success": bool(episode.get("success")),
               "replayed_reward": reward, "matches": (done and reward > 0) == bool(episode.get("success")),
               "actions": actions}
    (directory / "timeline.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def single_step(episode: dict) -> dict:
    """Adapt a one-action click-test record to the multi-step episode shape."""
    return {
        "task_seed": episode["seed"],
        "goal": episode.get("goal", "Click the button."),
        "success": episode["success"],
        "reward": episode["reward"],
        "steps": [{
            "text": episode["model_output"],
            "action": episode["parsed_action"],
            "valid_action": episode["valid_action"],
            "reward": episode["reward"],
            "observation": {},
        }],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task", default="book-flight-nodelay")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device-scale", type=float, default=4,
        help="Capture density for sharp frames; capture happens after each step, so it changes no outcome",
    )
    parser.add_argument(
        "--episodes-key",
        default="episodes",
        help="Dotted path to the episode list, e.g. trained_evaluation.stochastic.evaluation.episodes",
    )
    parser.add_argument("--vision", action="store_true", help="Screenshot-policy report with coordinate clicks")
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    for key in args.episodes_key.split("."):
        data = data[int(key)] if key.isdigit() else data[key]
    episodes = data if args.vision else [single_step(e) if "steps" not in e else e for e in data]
    if args.seeds:
        episodes = [e for e in episodes if e["task_seed"] in set(args.seeds)]
    root = ensure_miniwob_assets("artifacts/miniwob-plusplus")
    results = []
    with MiniwobAssetServer(root) as server, closing(
        _ScreenshotCapture.miniwob(args.task, miniwob_url=server.base_url, coordinate_actions=args.vision)
    ) as env:
        for episode in episodes:
            directory = args.output / f"seed-{episode['task_seed']}"
            if args.vision:
                summary = replay_vision(env, episode, directory, args.device_scale)
            else:
                upscale = 2 if args.device_scale <= 1 else 1
                summary = replay(env, episode, directory, upscale, args.device_scale)
            results.append({k: summary[k] for k in ("task_seed", "recorded_success", "replayed_reward", "matches")})
            print(json.dumps(results[-1]))
    (args.output / "index.json").write_text(json.dumps(results, indent=2) + "\n")
    if not all(r["matches"] for r in results):
        raise SystemExit("A replay did not reach its recorded outcome")


if __name__ == "__main__":
    main()
