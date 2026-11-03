"""Replay captions name the element a bid points at; frames trim empty page background."""

import pytest

from browser_control.replay import describe_call, element_labels, task_box

TREE = (
    "\t[19] textbox 'From:'\n\t[27] button 'Search'\n\t[40] link ''\n"
    "\t\t[36] listitem ''\n\t\t\tListMarker ''\n\t\t\tStaticText 'Hot Springs, AR (HOT)'\n"
    "\t\t[38] listitem ''"
)


def test_captions_name_the_target_element():
    labels = element_labels(TREE)
    fill = "<tool_call>\n<function=fill>\n<parameter=bid>\n19\n</parameter>\n<parameter=value>\nHOT\n</parameter>\n</function>\n</tool_call>"
    click = "<tool_call><function=click><parameter=bid>27</parameter></function></tool_call>"
    assert describe_call(fill, labels) == "fill textbox 'From:'  (value='HOT')"
    assert describe_call(click, labels) == "click button 'Search'"
    assert describe_call("<function=click><parameter=bid>99</parameter></function>", labels) == "click element 99"
    assert labels["40"] == "link"
    assert labels["36"] == "listitem 'Hot Springs, AR (HOT)'"
    assert labels["38"] == "listitem"
    assert describe_call("", labels) == "invalid tool call"
    assert describe_call("click('13')", labels) == "wrote \"click('13')\""


def test_task_box_trims_background_and_empty_instruction_bar():
    Image = pytest.importorskip("PIL.Image")
    page = Image.new("RGB", (100, 80), (85, 85, 85))
    page.paste(Image.new("RGB", (40, 50), (255, 255, 0)), (0, 0))
    page.paste(Image.new("RGB", (40, 30), (200, 220, 240)), (0, 20))
    assert task_box(page) == (0, 20, 40, 50)


def test_side_by_side_holds_the_shorter_episode_and_labels_outcomes():
    Image = pytest.importorskip("PIL.Image")
    from browser_control.replay import side_by_side_frames

    short = [Image.new("RGB", (48, 40), "red")]
    long = [Image.new("RGB", (48, 40), c) for c in ("blue", "green", "white")]
    frames = side_by_side_frames([short, long], ["Base", "Tuned"], [False, True])
    assert len(frames) == 3 and frames[0].width > 2 * 48
    with pytest.raises(ValueError):
        side_by_side_frames([short], ["Base", "Tuned"], [False])
