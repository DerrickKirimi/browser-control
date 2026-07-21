"""Decoder conditions and summaries for the Phase 5 checkpoint re-score."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from browser_control.evaluation import EvaluationReport, wilson_interval
from browser_control.transformers_policy import (
    CURRENT_EVALUATION_DECODER,
    DecoderConfig,
)

TRAINING_DECODER = DecoderConfig(
    name="training",
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=32,
)
NO_REPETITION_PENALTY_ABLATION = DecoderConfig(
    name="evaluation-no-repetition-penalty",
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.0,
    max_new_tokens=16,
)
GREEDY_DECODER = DecoderConfig(
    name="greedy",
    do_sample=False,
    max_new_tokens=32,
)
GATE_A_DECODERS = (
    TRAINING_DECODER,
    NO_REPETITION_PENALTY_ABLATION,
    CURRENT_EVALUATION_DECODER,
    GREEDY_DECODER,
)


@dataclass(frozen=True, slots=True)
class RescoreSummary:
    decoder: dict[str, object]
    samples: int
    successes: int
    success_rate: float
    wilson_95: tuple[float, float] | None
    evaluation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_rescore(
    decoder: DecoderConfig, report: EvaluationReport
) -> RescoreSummary:
    """Summarize one fixed-prompt decoder condition without claiming generalization."""

    samples = len(report.episodes)
    successes = sum(episode.success for episode in report.episodes)
    interval = wilson_interval(successes, samples) if decoder.do_sample else None
    return RescoreSummary(
        decoder=asdict(decoder),
        samples=samples,
        successes=successes,
        success_rate=report.success_rate,
        wilson_95=interval,
        evaluation=report.to_dict(),
    )
