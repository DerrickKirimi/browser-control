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
