from browser_control.evaluation import evaluate_click_test
from browser_control.rescoring import (
    GREEDY_DECODER,
    TRAINING_DECODER,
    summarize_rescore,
)
from browser_control.testing import FakeClickTestEnvironment


class SuccessfulPolicy:
    def generate(self, messages, *, seed):
        from browser_control.evaluation import PolicyOutput

        return PolicyOutput("click('13')", 0.01, 10, 3)


def test_stochastic_rescore_includes_interval() -> None:
    report = evaluate_click_test(
        FakeClickTestEnvironment, SuccessfulPolicy(), seeds=range(50)
    )

    summary = summarize_rescore(TRAINING_DECODER, report)

    assert summary.samples == 50
    assert summary.successes == 50
    assert summary.wilson_95 is not None
    assert summary.to_dict()["decoder"]["name"] == "training"


def test_greedy_rescore_has_no_sampling_interval() -> None:
    report = evaluate_click_test(FakeClickTestEnvironment, SuccessfulPolicy(), seeds=[0])

    assert summarize_rescore(GREEDY_DECODER, report).wilson_95 is None
