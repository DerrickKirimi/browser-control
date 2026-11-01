"""Pinned vision model identity and screenshot-only native coordinate prompts."""

from __future__ import annotations

import io

from browser_control.vision_browser import VisionObservation

MODEL_ID = "LiquidAI/LFM3-VL-2B"
MODEL_REVISION = "ba44bd51db20a2753f28c273a1571ccaa210d954"
COORDINATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click a screenshot location using normalized coordinates0–1000.",
            "parameters": {
                "type": "object",
                "properties": {
                    axis: {"type": "integer", "minimum": 0, "maximum": 1000}
                    for axis in ("x", "y")
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    }
]


def build_vision_messages(observation: VisionObservation):
    from PIL import Image

    image = Image.open(io.BytesIO(observation.screenshot_png))
    image.load()
    if image.mode != "RGB" or image.size != (observation.width, observation.height):
        raise ValueError("Screenshot dimensions/mode differ from the observation")
    text = observation.goal
    if observation.last_action_error:
        text += "\nLast action error: " + observation.last_action_error
    return [
        {
            "role": "system",
            "content": "Complete the browser task from the screenshot. Call exactly one provided tool. Coordinates x/y are integers0–1000, spanning the entire screenshot from left/top to right/bottom. Return only the tool call, without reasoning or prose.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        },
    ]
