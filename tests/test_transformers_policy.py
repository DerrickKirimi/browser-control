from __future__ import annotations

import pytest

from browser_control.transformers_policy import DecoderConfig


def test_sampling_decoder_emits_only_declared_generation_arguments() -> None:
    decoder = DecoderConfig(
        name="training",
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        max_new_tokens=32,
    )

    assert decoder.generation_kwargs() == {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_new_tokens": 32,
    }


def test_greedy_decoder_rejects_sampling_arguments() -> None:
    with pytest.raises(ValueError, match="greedy"):
        DecoderConfig(
            name="invalid-greedy",
            do_sample=False,
            temperature=1.0,
            max_new_tokens=16,
        )


def test_decoder_rejects_two_probability_cutoffs() -> None:
    with pytest.raises(ValueError, match="top_p or min_p"):
        DecoderConfig(
            name="ambiguous",
            do_sample=True,
            top_p=0.95,
            min_p=0.15,
            max_new_tokens=16,
        )
