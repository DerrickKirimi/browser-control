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
pixels are zeroed. The frozen policy wrapper additionally checks bounded cached
generation, every next-token forward's logits, frozen weights, raw prompt/completion
IDs and rejection of deliberately injected nonfinite logits. It records
package/source/input hashes. It does not load
pretrained weights or validate FP16/T4 inference, RL masks or vision GRPO.

## Frozen one-click evaluation

Use `requirements-lfm3-vision.txt` in a separate Python3.12 runtime. It contains
inference pins; vision GRPO is not validated by installing it. The loader requires
a T4 and uses FP16 with SDPA, the pinned processor/model revision, disabled thinking,
a2048token prompt budget and a128token completion cap.

```bash
python scripts/evaluate_lfm3_vision.py --output frozen.json \
  --decoder greedy --image-condition screenshot
```

Tasks are `click-color`, `click-shape` and `click-shades`, with development seeds
30–59 by default. `--decoder sampling` explicitly uses temperature1/top_p.95;
checkpoint top-k and repetition penalties are cleared. `--image-condition white`
replaces pixels with a white image while keeping the task instruction and image
dimensions. Use the same tasks, seeds and generation rule for both image conditions.

Each episode records source/input image hashes and PNGs, actual prompt/completion
IDs, image grid, finite-logit checks, normalized/rendered coordinates and native
terminal reward. Invalid outputs are counted without repair or an environment
action. Numerical failures abort the run. Per-task success counts and Wilson
intervals are recomputed from native outcomes. This remains development evaluation;
neither the CLI nor the CPU contract establishes a pretrained vision or RL result.
