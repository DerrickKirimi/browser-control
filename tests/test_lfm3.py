"""Validate the native-call execution boundary without loading model weights."""

import pytest

from browser_control.actions import InvalidActionError
from browser_control.evaluation import PolicyOutput, evaluate_click_test
from browser_control.lfm3 import build_lfm3_messages, parse_lfm3_action
from browser_control.testing import FakeClickTestEnvironment

CLICK = "<tool_call>\n<function=click>\n<parameter=bid>\n13\n</parameter>\n</function>\n</tool_call>"


def test_native_call_passes_existing_action_validation():
    assert parse_lfm3_action(CLICK).render() == "click('13')"
    assert (
        parse_lfm3_action("<tool_call><function=noop></function></tool_call>").render()
        == "noop()"
    )


@pytest.mark.parametrize(
    "output",
    [
        "",
        "click('13')",
        CLICK + CLICK,
        "Sure: " + CLICK,
        CLICK.replace("13", "13');__import__('os')"),
        CLICK.replace("parameter=bid", "parameter=other"),
        CLICK.replace("function=click", "function=fill"),
        CLICK.replace("</function>", "<parameter=extra>x</parameter></function>"),
        CLICK.replace("13", ""),
        CLICK[:-12],
        "<tool_call><function=noop><parameter=bid>13</parameter></function></tool_call>",
    ],
)
def test_invalid_native_calls_are_rejected(output):
    with pytest.raises(InvalidActionError):
        parse_lfm3_action(output)


def test_native_prompt_has_no_concrete_answer_example():
    env = FakeClickTestEnvironment(button_id="random_82")
    messages = build_lfm3_messages(env.reset(seed=23).observation)
    assert "random_82" not in messages[0]["content"]
    assert "13" not in messages[0]["content"]
    assert "random_82" in messages[1]["content"]


def test_native_evaluation_retains_invalid_raw_output_and_does_not_step():
    class Policy:
        def generate(self, messages, *, seed):
            return PolicyOutput(CLICK if seed == 23 else "bad XML", 0.0, 10, 20)

    report = evaluate_click_test(
        FakeClickTestEnvironment,
        Policy(),
        [23, 24],
        message_builder=build_lfm3_messages,
        action_parser=parse_lfm3_action,
    )
    assert report.success_rate == 0.5
    assert report.episodes[0].parsed_action == "click('13')"
    assert report.episodes[1].model_output == "bad XML"
    assert report.episodes[1].environment_steps == 0


def gate_conditions(native_successes=48):
    return [
        {
            "interface": interface,
            "decoder": {"name": mode},
            "seed": seed,
            "numerical_check": {"finite": True},
            "evaluation": {
                "episodes": [
                    {"success": index < (native_successes if mode == "sampling" else 1)}
                    for index in range(50 if mode == "sampling" else 1)
                ]
            },
        }
        for interface in ("strict-text", "native-tools")
        for mode in ("greedy", "sampling")
        for seed in (23, 47, 71)
    ]


def test_frozen_gate_requires_each_native_cell_to_pass():
    from browser_control.lfm3 import frozen_gate

    conditions = gate_conditions()
    assert frozen_gate(conditions)["passed"]
    conditions[-1]["evaluation"]["episodes"][47]["success"] = False
    assert frozen_gate(conditions) == {
        "eligible": True,
        "passed": False,
        "decision": "inspect-format-length-and-numerics",
    }


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "short", "nonfinite", "greedy"]
)
def test_frozen_gate_rejects_incomplete_and_invalid_evidence(change):
    from browser_control.lfm3 import frozen_gate

    conditions = gate_conditions()
    if change == "missing":
        conditions.pop()
    elif change == "duplicate":
        conditions[-1] = conditions[-2]
    elif change == "short":
        conditions[-1]["evaluation"]["episodes"].pop()
    elif change == "nonfinite":
        conditions[-1]["numerical_check"]["finite"] = False
    else:
        conditions[6]["evaluation"]["episodes"][0]["success"] = False
    assert not frozen_gate(conditions)["passed"]


def test_model_loading_replaces_checkpoint_generation_filters(monkeypatch):
    import sys
    from types import SimpleNamespace

    from browser_control.lfm3 import LFM3Policy
    from browser_control.transformers_policy import DecoderConfig

    model = SimpleNamespace(
        config=SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=0),
        generation_config=SimpleNamespace(repetition_penalty=1.1, top_k=50),
    )
    model.to = lambda device: model
    model.eval = lambda: model
    tokenizer = object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model),
            GenerationConfig=SimpleNamespace,
        ),
    )
    monkeypatch.setattr(
        "browser_control.lfm3.inspect_cuda_device",
        lambda torch: SimpleNamespace(training_dtype="float16"),
    )
    policy = LFM3Policy.load(DecoderConfig("greedy", False, 128), "strict-text")
    assert policy.model.generation_config.repetition_penalty == 1.0
    assert policy.model.generation_config.top_k == 0
    assert policy.model.generation_config.eos_token_id == 2
