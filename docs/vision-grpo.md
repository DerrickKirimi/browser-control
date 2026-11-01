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
