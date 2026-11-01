# Vision environment interface

`browser_control.vision_browser.VisionBrowserGymEnvironment.miniwob("click-color")`
provides task instructions, RGB PNG screenshots, image dimensions and execution
errors. Accessibility trees and browser element IDs are excluded from its policy
observation. The environment extra, Pillow, NumPy and Chromium are required.

`parse_coordinate_click` accepts exactly one native XML `click` with integer
parameters `x` and `y` in0–1000. Coordinates map to the inclusive screenshot pixel
range. The environment uses BrowserGym's coordinate action set; BrowserGym maps
its enlarged screenshot pixels back to page coordinates. Do not apply an extra
scale conversion or change the benchmark viewport.

```xml
<tool_call><function=click><parameter=x>500</parameter><parameter=y>500</parameter></function></tool_call>
```

This is environment transport for the subsequent vision stage. It does not load
a vision model or establish frozen-policy or GRPO performance.

## Processor and architecture check

The vision identity and native tool prompt live in `browser_control.lfm3_vision`.
They pin `LiquidAI/LFM3-VL-2B` to revision
`ba44bd51db20a2753f28c273a1571ccaa210d954`.
Use a separate runtime with Transformers5.21.0, Torch2.14.1 and
Torchvision0.29.1; the text runtime retains its own pins.

Download only the pinned model's configuration, tokenizer, processor, Python files
and `chat_template.jinja`, then run:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/check_lfm3_vision_contract.py \
  --model-source PATH --screenshot SCREENSHOT.png --goal 'TASK INSTRUCTION' \
  --output contract.json
```

This uses tiny random float32 weights and the actual processor. It checks image
token alignment, finite logits/backward gradients and changed logits when processed
pixels are zeroed. It records package/source/input hashes. It does not load
pretrained weights or validate FP16/T4 inference, RL masks or vision GRPO.
