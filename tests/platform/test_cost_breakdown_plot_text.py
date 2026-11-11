"""Segment order and bar wording of the cost-breakdown figure, as pure functions."""

from __future__ import annotations

import pytest

from neuralls.platform.reporting.cost_breakdown_plot import (
    CostSegment,
    cost_bar_label,
    cost_bar_text,
    cost_segments,
)

COMPLETE_ROW = {
    "generation_status": "known",
    "training_status": "known",
    "fit_origin": "job",
    "total_s": 102.5,
    "total_status": "complete",
}


def test_the_bar_stacks_offline_stages_before_online_ones() -> None:
    assert cost_segments() == (
        CostSegment.GENERATION,
        CostSegment.TRAINING,
        CostSegment.SETUP,
        CostSegment.SOLVE,
    )


@pytest.mark.parametrize(
    ("generation", "training", "expected"),
    [
        ("known", "unavailable", "total unknown: offline fit / training unavailable"),
        ("unavailable", "known", "total unknown: generation unavailable"),
        (
            "unavailable",
            "unavailable",
            "total unknown: generation and offline fit / training unavailable",
        ),
    ],
)
def test_a_total_with_an_unavailable_stage_names_the_stages_and_states_no_number(
    generation: str, training: str, expected: str
) -> None:
    row = {
        "generation_status": generation,
        "training_status": training,
        "total_s": None,
        "total_status": "offline_unavailable",
    }

    assert cost_bar_label(row) == expected


@pytest.mark.parametrize(
    ("fit_origin", "expected"),
    [
        ("job", "102.5 s\nfit job"),
        ("inline", "102.5 s\nfit inline, counted in setup"),
        ("none", "102.5 s"),
    ],
)
def test_the_text_above_a_bar_adds_the_fit_origin_to_the_total(
    fit_origin: str, expected: str
) -> None:
    assert cost_bar_text({**COMPLETE_ROW, "fit_origin": fit_origin}) == expected


def test_an_unknown_total_keeps_its_note_before_the_fit_origin() -> None:
    row = {
        "generation_status": "known",
        "training_status": "unavailable",
        "fit_origin": "job",
        "total_s": None,
        "total_status": "offline_unavailable",
    }

    assert cost_bar_text(row) == "total unknown: offline fit / training unavailable\nfit job"
