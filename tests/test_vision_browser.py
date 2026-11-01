"""Visual actions are bounded; policy observations contain pixels without DOM."""

import io
from dataclasses import asdict

import pytest

from browser_control.actions import InvalidActionError
from browser_control.lfm3_vision import COORDINATE_TOOLS, build_vision_messages
from browser_control.vision_browser import (
    CoordinateClick,
    VisionBrowserGymEnvironment,
    parse_coordinate_click,
)


def xml(x="500", y="500"):
    return f"<tool_call><function=click><parameter=x>{x}</parameter><parameter=y>{y}</parameter></function></tool_call>"


def test_normalized_coordinates_include_both_pixel_edges():
    assert parse_coordinate_click(xml("0", "1000")).render(160, 210) == "mouse_click(0, 209)"
    assert parse_coordinate_click(xml()).render(160, 210) == "mouse_click(80, 104)"
    assert CoordinateClick(1000, 0).render(160, 210) == "mouse_click(159, 0)"


@pytest.mark.parametrize(
    "output",
    [
        xml("-1"), xml("1001"), xml("0.5"), xml("nan"), xml("True"),
        xml() + xml(), "reasoning" + xml(), xml().replace("parameter=y", "parameter=x"),
        xml().replace("parameter=y", "parameter=bid"),
        xml("__import__('os').system('echo unsafe')"),
    ],
)
def test_invalid_coordinate_output_is_never_repaired(output):
    with pytest.raises(InvalidActionError):
        parse_coordinate_click(output)


def test_programmatic_actions_and_viewports_are_also_validated():
    for value in (-1, 1001, True, 0.5):
        with pytest.raises(InvalidActionError):
            CoordinateClick(value, 500)
    for dimensions in ((0, 10), (10, -1), (True, 10)):
        with pytest.raises(ValueError):
            CoordinateClick(500, 500).render(*dimensions)


class Gym:
    def __init__(self, pixels):
        self.pixels = pixels
        self.actions = []
        self.closed = False

    def observation(self):
        return {
            "goal": "Click the red shape.",
            "axtree_txt": "SECRET DOM ANSWER",
            "screenshot": self.pixels,
        }

    def reset(self, seed=None):
        return self.observation(), {}

    def step(self, action):
        self.actions.append(action)
        self.pixels = self.pixels.copy()
        self.pixels[0, 0] = [0, 255, 0]
        return self.observation(), 1.0, True, False, {}

    def close(self):
        self.closed = True


def test_pixels_are_preserved_without_exposing_dom_and_actions_follow_lifecycle():
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    pixels = np.zeros((210, 160, 3), dtype=np.uint8)
    pixels[0, 0] = [255, 0, 0]
    gym = Gym(pixels)
    env = VisionBrowserGymEnvironment(gym)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(CoordinateClick(0, 0))
    before = env.reset(seed=23)
    assert set(asdict(before.observation)) == {
        "goal", "screenshot_png", "width", "height", "last_action_error"
    }
    assert b"SECRET" not in before.observation.screenshot_png
    image = Image.open(io.BytesIO(before.observation.screenshot_png))
    assert image.size == (160, 210)
    assert np.array_equal(np.array(image), pixels)
    after = env.step(parse_coordinate_click(xml("1000", "1000")))
    assert gym.actions == ["mouse_click(159, 209)"]
    assert after.done and after.reward == 1.0
    assert before.observation.screenshot_png != after.observation.screenshot_png
    with pytest.raises(RuntimeError, match="done"):
        env.step(CoordinateClick(0, 0))
    env.close()
    assert gym.closed


def test_missing_or_non_rgb_pixels_are_rejected():
    np = pytest.importorskip("numpy")
    for pixels in (None, np.zeros((5, 5)), np.zeros((5, 5, 4), dtype=np.uint8)):
        env = VisionBrowserGymEnvironment(Gym(pixels))
        with pytest.raises(ValueError, match="RGB uint8"):
            env.reset(seed=23)


def test_vision_prompt_uses_actual_pixels_goal_and_declared_coordinate_tools():
    np = pytest.importorskip("numpy")
    env = VisionBrowserGymEnvironment(Gym(np.zeros((10, 20, 3), dtype=np.uint8)))
    observation = env.reset(seed=23).observation
    messages = build_vision_messages(observation)
    assert [item["role"] for item in messages] == ["system", "user"]
    content = messages[1]["content"]
    assert content[0]["image"].size == (20, 10)
    assert content[1] == {"type": "text", "text": observation.goal}
    assert "SECRET" not in repr(messages)
    parameters = COORDINATE_TOOLS[0]["function"]["parameters"]
    assert parameters["required"] == ["x", "y"]
    assert not parameters["additionalProperties"]
    from dataclasses import replace

    with pytest.raises(ValueError, match="dimensions/mode"):
        build_vision_messages(replace(observation, width=21))
