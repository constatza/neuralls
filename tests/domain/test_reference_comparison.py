"""Temporary observed-reference comparison until explicit configuration lands."""

from __future__ import annotations

import pytest

from neuralls.domain.benchmarking.reference_comparison import compare_to_reference
from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome
from neuralls.shared.types import MatrixId

SEED = 11


def _solved(time: float) -> Solved:
    return Solved(MatrixId(1), 1, solve_s=time, amortized_online_s=time)


@pytest.fixture
def outcomes() -> dict[str, dict[str, SystemOutcome]]:
    return {
        "amg": {"s0": _solved(4.0), "s1": _solved(6.0), "s2": _solved(8.0)},
        "jacobi": {
            "s0": _solved(2.0),
            "s1": _solved(3.0),
            "s2": Failed(MatrixId(1), 100, "limit"),
        },
        "never": {
            "s0": Failed(MatrixId(1), 100, "limit"),
            "s1": Failed(MatrixId(1), 100, "limit"),
            "s2": Failed(MatrixId(1), 100, "limit"),
        },
    }


def test_reference_is_skipped_but_every_other_method_is_compared(outcomes) -> None:
    comparisons = compare_to_reference(outcomes, seed=SEED)

    assert set(comparisons) == {"jacobi", "never"}
    assert comparisons["jacobi"].reference == "amg"
    assert comparisons["never"].paired.n_reference_only == 3


def test_paired_part_uses_reference_over_method_ratio(outcomes) -> None:
    comparison = compare_to_reference(outcomes, seed=SEED)["jacobi"]

    assert comparison.paired.speed_ratios == (2.0, 2.0)


def test_empty_group_yields_no_comparisons() -> None:
    assert compare_to_reference({}, seed=SEED) == {}
