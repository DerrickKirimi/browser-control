"""Multi-step action safety, history and terminal/horizon contracts."""

import ast
from types import SimpleNamespace

import pytest

from browser_control.actions import InvalidActionError
from browser_control.book_flight import parse_book_flight_action
from browser_control.book_flight_evaluation import collect_book_flight_episode
from browser_control.environment import BrowserObservation, BrowserStep
from browser_control.evaluation import PolicyOutput
from browser_control.local_browsergym import LocalBrowserGymEnvironment


def xml(name, **parameters):
    body = "".join(f"<parameter={k}>{v}</parameter>" for k, v in parameters.items())
    return f"<tool_call><function={name}>{body}</function></tool_call>"


def test_fill_value_is_data_and_autocomplete_is_enabled():
    value = "x'); __import__('os').system('bad') #"
    action = parse_book_flight_action(xml("fill", bid="abc_12", value=value))
    call = ast.parse(action.render(), mode="eval").body
    assert isinstance(call, ast.Call)
    assert [ast.literal_eval(arg) for arg in call.args] == ["abc_12", value, True]
    assert action.message()["tool_calls"][0]["function"]["arguments"]["value"] == value


@pytest.mark.parametrize(
    "output",
    [
        xml("fill", bid="1", value="x", extra="y"),
        xml("fill", bid="1"),
        xml("press", bid="1", key="Control+W"),
        xml("click", bid="1');bad()"),
        xml("fill", bid="1", value="a" * 257),
        xml("fill", bid="1", value="a\nb"),
        xml("eval", code="bad"),
        xml("noop") + xml("click", bid="1"),
        "Sure: " + xml("click", bid="1"),
        xml("click", bid="1").replace(
            "</function>", "<parameter=bid>2</parameter></function>"
        ),
    ],
)
def test_invalid_actions_are_rejected(output):
    with pytest.raises(InvalidActionError):
        parse_book_flight_action(output)


class Environment:
    def __init__(self, terminal=True):
        self.count = 0
        self.terminal = terminal

    def reset(self, *, seed):
        self.count = 0
        return BrowserStep(
            BrowserObservation("book a flight", "[1] textbox"), 0.0, False
        )

    def step(self, action):
        self.count += 1
        done = self.terminal and self.count == 2
        assert self.count <= 2 or not self.terminal
        return BrowserStep(
            BrowserObservation(
                "book a flight", "[2] button", "readonly" if self.count == 1 else None
            ),
            1.0 if done else 0.0,
            done,
        )


def test_reported_action_errors_are_context_and_can_be_recovered():
    seen = []

    def generate(messages, *, seed):
        seen.append((list(messages), seed))
        return PolicyOutput(xml("click", bid="1"), 0.1, 10, 20)

    env = Environment()
    result = collect_book_flight_episode(env, SimpleNamespace(generate=generate), 20)
    assert result["success"] and result["success_after_action_error"]
    assert result["environment_steps"] == 2
    assert seen[1][0][-1]["role"] == "tool"
    assert "readonly" in seen[1][0][-1]["content"]
    assert seen[1][0][-2]["tool_calls"][0]["function"]["name"] == "click"
    assert [seed for _, seed in seen] == [320, 321]


def test_invalid_call_ends_episode_without_an_environment_step():
    policy = SimpleNamespace(
        generate=lambda *a, **k: PolicyOutput("bad XML", 0.1, 10, 20)
    )
    env = Environment()
    result = collect_book_flight_episode(env, policy, 20)
    assert result["stop_reason"] == "invalid-action"
    assert env.count == 0 and result["environment_steps"] == 0
    assert result["steps"][0]["text"] == "bad XML"


def test_horizon_counts_model_turns_and_bounds_noop_loops():
    policy = SimpleNamespace(
        generate=lambda *a, **k: PolicyOutput(xml("noop"), 0.1, 10, 20)
    )
    result = collect_book_flight_episode(
        Environment(terminal=False), policy, 20, max_steps=3
    )
    assert not result["success"] and result["stop_reason"] == "horizon"
    assert result["model_turns"] == result["environment_steps"] == 3


def test_browser_error_text_is_preserved_for_recovery():
    step = LocalBrowserGymEnvironment._normalize(
        observation={"last_action_error": "TimeoutError: readonly field"},
        reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert step.observation.last_action_error == "TimeoutError: readonly field"


def test_context_limit_refuses_truncation_before_copying_to_gpu():
    from browser_control.lfm3 import LFM3Policy, PromptTooLongError
    from browser_control.transformers_policy import DecoderConfig

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *a, **k: {
            "input_ids": SimpleNamespace(shape=(1, 4)),
        }
    )
    policy = LFM3Policy(
        None,
        tokenizer,
        None,
        DecoderConfig("greedy", False, 128),
        "native-tools",
        max_prompt_tokens=3,
    )
    with pytest.raises(PromptTooLongError) as raised:
        policy.encode([{"role": "user", "content": "long observation"}])
    assert raised.value.tokens == 4 and raised.value.limit == 3
