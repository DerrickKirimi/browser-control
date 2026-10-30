"""Paired ID evaluation must not accidentally enable training augmentation."""

import runpy
from pathlib import Path

sft_command = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/colab_lfm3_frozen.py")
)["sft_command"]


def test_id_evaluation_does_not_enable_id_augmentation():
    command = sft_command("python", "output", evaluate_permuted=True)
    assert "--permuted-probes" in command
    assert "--id-permutation-seed" not in command
    assert "None" not in command


def test_id_training_enables_matched_evaluation():
    command = sft_command("python", "output", id_permutation_seed=23)
    assert command[command.index("--id-permutation-seed") + 1] == "23"
    assert "--permuted-probes" in command


def test_focused_continuation_has_explicit_adapter_skills_and_pass_budget():
    command = sft_command(
        "python",
        "output",
        updates=64,
        passes=4,
        initial_adapter="adapter",
        focus_skills=["choose_date", "book_flight"],
        evaluate_permuted=True,
    )
    assert command[command.index("--updates") + 1] == "64"
    assert command[command.index("--passes") + 1] == "4"
    assert command[command.index("--initial-adapter") + 1] == "adapter"
    assert "--seen-training-fit" in command
    assert "--id-permutation-seed" not in command
    assert command[
        command.index("--focus-skills") + 1 : command.index("--seen-training-fit")
    ] == ["choose_date", "book_flight"]
