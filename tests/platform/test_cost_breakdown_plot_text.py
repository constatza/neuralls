"""Segment order and bar wording of the cost-breakdown figure, as pure functions."""

from __future__ import annotations

import pytest

from neuralls.platform.reporting.cost_breakdown_plot import (
    CostSegment,
    cost_bar_label,
    cost_segments,
)

COMPLETE_ROW = {
    "generation_status": "known",
    "training_status": "known",
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


def test_each_segment_is_named_by_its_table_column() -> None:
    assert [segment.value for segment in cost_segments()] == [
        "generation_s",
        "training_s",
        "allocated_setup_s",
        "solve_total_s",
    ]


def test_a_complete_total_is_stated_in_seconds() -> None:
    assert cost_bar_label(COMPLETE_ROW) == "102.5 s"


@pytest.mark.parametrize(
    ("generation", "training", "expected"),
    [
        ("known", "unavailable", "total unknown: training unavailable"),
        ("unavailable", "known", "total unknown: generation unavailable"),
        ("unavailable", "unavailable", "total unknown: generation and training unavailable"),
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
