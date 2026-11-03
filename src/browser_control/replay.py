"""Annotated replay frames for recorded MiniWoB episodes."""

from __future__ import annotations

import io
import re
import textwrap

PANEL = 132
SCALE = 2
BID_LINE = re.compile(r"\[(\d+)\]\s+(\w+)\s+'([^']*)'")
CALL = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
PARAMETER = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def element_labels(tree: str) -> dict[str, str]:
    """Map bids to "role 'name'"; an unnamed element takes the first text nested inside it."""
    labels, lines = {}, tree.split("\n")
    for index, line in enumerate(lines):
        match = BID_LINE.search(line)
        if match is None:
            continue
        bid, role, name = match.groups()
        depth = len(line) - len(line.lstrip("\t"))
        for child in lines[index + 1 :]:
            if name or len(child) - len(child.lstrip("\t")) <= depth or BID_LINE.search(child):
                break
            text = re.search(r"StaticText '([^']+)'", child)
            if text:
                name = text.group(1)
        labels[bid] = f"{role} '{name}'" if name else role
    return labels


def describe_call(text: str, labels: dict[str, str]) -> str:
    """Render the model's tool call, naming the element its bid points at."""
    match = CALL.search(text or "")
    if match is None:
        plain = " ".join((text or "").split())
        return f"wrote {plain[:70]!r}" if plain else "invalid tool call"
    name, body = match.group(1), match.group(2)
    parameters = dict(PARAMETER.findall(body))
    bid = parameters.pop("bid", None)
    target = f" {labels.get(bid, 'element ' + bid)}" if bid else ""
    rest = ", ".join(f"{key}={value!r}" for key, value in parameters.items())
    return f"{name}{target}" + (f"  ({rest})" if rest else "")


def task_box(page):
    """Bound the MiniWoB task area by trimming the flat page background on the right and bottom."""
    from PIL import Image, ImageChops

    background = Image.new("RGB", page.size, page.getpixel((page.width - 1, page.height - 1)))
    box = ImageChops.difference(page, background).getbbox()
    right, bottom = (box[2], box[3]) if box else page.size
    top = 0
    while top < bottom - 1 and all(page.getpixel((x, top)) == (255, 255, 0) for x in range(0, right, 4)):
        top += 1
    return (0, top, right, bottom)


def annotate(png: bytes, goal: str, caption: str, status: str, upscale: int = SCALE):
    """Crop to the task, enlarge by `upscale` (1 for screenshots already taken at high density),
    and add a caption panel sized to the frame."""
    from PIL import Image, ImageDraw, ImageFont

    page = Image.open(io.BytesIO(png)).convert("RGB")
    page = page.crop(task_box(page))
    if upscale != 1:
        page = page.resize((page.width * upscale, page.height * upscale), Image.NEAREST)
    unit = page.width / 480
    panel = round(PANEL * unit)
    frame = Image.new("RGB", (page.width, page.height + panel), (17, 17, 23))
    frame.paste(page, (0, 0))
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", round(18 * unit))
        small = ImageFont.truetype("DejaVuSans.ttf", round(15 * unit))
    except OSError:
        font = small = ImageFont.load_default()
    margin, chars = round(14 * unit), 48
    y = page.height + round(10 * unit)
    for line in textwrap.wrap(goal, chars + 12)[:2]:
        draw.text((margin, y), line, fill=(170, 170, 185), font=small)
        y += round(20 * unit)
    for line in textwrap.wrap(caption, chars)[:2]:
        draw.text((margin, y + round(4 * unit)), line, fill=(255, 255, 255), font=font)
        y += round(24 * unit)
    draw.text((margin, page.height + panel - round(26 * unit)), status, fill=(120, 220, 140), font=small)
    return frame


def side_by_side_frames(columns, labels, outcomes, title=""):
    """Frame-by-frame composite of several replays of one task, with a labelled header per column.

    Shorter columns hold their last frame. The header shows each policy's recorded outcome.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not columns or any(not frames for frames in columns) or not len(columns) == len(labels) == len(outcomes):
        raise ValueError("One nonempty frame list, label and outcome per column")
    width = max(f.width for frames in columns for f in frames)
    height = max(f.height for frames in columns for f in frames)
    unit = width / 480
    gap, header = round(16 * unit), round((74 if title else 50) * unit)
    try:
        bold = ImageFont.truetype("DejaVuSans-Bold.ttf", round(19 * unit))
        small = ImageFont.truetype("DejaVuSans.ttf", round(15 * unit))
    except OSError:
        bold = small = ImageFont.load_default()
    composites = []
    for index in range(max(len(frames) for frames in columns)):
        canvas = Image.new("RGB", (len(columns) * width + (len(columns) + 1) * gap, header + height + gap), (245, 246, 250))
        draw = ImageDraw.Draw(canvas)
        if title:
            draw.text((gap, round(8 * unit)), title, fill=(60, 64, 80), font=small)
        for column, (frames, label, success) in enumerate(zip(columns, labels, outcomes)):
            frame = frames[min(index, len(frames) - 1)]
            x = gap + column * (width + gap)
            done = index >= len(frames) - 1
            tag = ("  ✓ success" if success else "  ✗ failure") if done else ""
            draw.text((x, header - round(32 * unit)), label + tag, font=bold,
                      fill=(20, 120, 60) if done and success else (170, 40, 40) if done else (25, 27, 38))
            canvas.paste(frame, (x, header))
        composites.append(canvas)
    return composites
