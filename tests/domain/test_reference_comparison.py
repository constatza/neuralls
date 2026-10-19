"""Group-level comparison against the reference method (``domain.benchmarking.reference_comparison``)."""

from __future__ import annotations

import pytest

from neuralls.domain.benchmarking.paired import compare_paired, paired_points
from neuralls.domain.benchmarking.reference_comparison import compare_to_reference
from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.sharing import SetupSharingPolicy

SEED = 11


def _projection(name: str) -> ProjectedComparisonResult:
    return ProjectedComparisonResult(
        preconditioner_name=name,
        n_systems=3,
        n_converged=3,
        setup_sharing=SetupSharingPolicy(share_setup=True, share_generation=True),
        setup_cost_total_seconds=3.0,
        generation_cost_total_seconds=3.0,
        total_solve_time_seconds=6.0,
        total_iterations=30,
    )


@pytest.fixture
def outcomes() -> dict[str, dict[str, SystemOutcome]]:
    """``amg`` solves all three systems (reference), ``jacobi`` two, ``never`` none."""
    return {
        "amg": {s: Solved(1, t) for s, t in (("s0", 4.0), ("s1", 6.0), ("s2", 8.0))},
        "jacobi": {"s0": Solved(1, 2.0), "s1": Solved(1, 3.0), "s2": Failed()},
        "never": {"s0": Failed(), "s1": Failed(), "s2": Failed()},
    }


def test_reference_is_skipped_and_method_without_projection_is_skipped(outcomes) -> None:
    projected = {"amg": _projection("amg"), "jacobi": _projection("jacobi")}

    comparisons = compare_to_reference(outcomes, projected, seed=SEED)

    assert set(comparisons) == {"jacobi"}
    assert comparisons["jacobi"].reference == "amg"


def test_paired_part_uses_the_reference_and_method_outcomes(outcomes) -> None:
    projected = {"amg": _projection("amg"), "jacobi": _projection("jacobi")}

    comparison = compare_to_reference(outcomes, projected, seed=SEED)["jacobi"]

    expected = compare_paired(paired_points(outcomes["amg"], outcomes["jacobi"]), seed=SEED)
    assert comparison.paired == expected
    assert comparison.paired is not None
    assert comparison.paired.speedups == (2.0, 2.0)


def test_missing_reference_projection_yields_no_comparisons(outcomes) -> None:
    assert compare_to_reference(outcomes, {"jacobi": _projection("jacobi")}, seed=SEED) == {}


def test_empty_group_yields_no_comparisons() -> None:
    assert compare_to_reference({}, {}, seed=SEED) == {}
