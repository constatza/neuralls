"""Explicit references never come from observed outcomes."""

from __future__ import annotations

from neuralls.domain.benchmarking.reference_comparison import compare_to_reference
from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    NoReferenceConfigured,
    ReferenceUnavailable,
)
from neuralls.domain.solver.models.outcomes import Failed, Solved
from neuralls.shared.types import MatrixId, PreconditionerName


def test_configured_reference_preserves_asymmetric_success_counts() -> None:
    outcomes = {
        PreconditionerName("amg"): {
            "s0": Failed(MatrixId(1), 20, "limit"),
            "s1": Solved(MatrixId(1), 2, 4.0, 4.0),
            "s2": Solved(MatrixId(1), 2, 4.0, 4.0),
        },
        PreconditionerName("jacobi"): {
            "s0": Solved(MatrixId(1), 2, 2.0, 2.0),
            "s1": Solved(MatrixId(1), 2, 2.0, 2.0),
            "s2": Failed(MatrixId(1), 20, "limit"),
        },
    }

    comparisons = compare_to_reference(
        outcomes,
        selection=ConfiguredReference(PreconditionerName("amg")),
        seed=11,
    )

    paired = comparisons[PreconditionerName("jacobi")].paired
    assert paired.n_reference_only == 1
    assert paired.n_method_only == 1
    assert paired.n_both_solved == 1
    assert paired.speed_ratios == (2.0,)
    assert comparisons[PreconditionerName("jacobi")].reference == PreconditionerName("amg")


def test_no_reference_and_unavailable_reference_have_no_comparisons() -> None:
    outcomes = {
        PreconditionerName("amg"): {
            "s0": Solved(MatrixId(1), 2, 4.0, 4.0),
        },
        PreconditionerName("jacobi"): {
            "s0": Solved(MatrixId(1), 2, 2.0, 2.0),
        },
    }

    assert compare_to_reference(outcomes, selection=NoReferenceConfigured(), seed=11) == {}
    assert (
        compare_to_reference(
            outcomes,
            selection=ReferenceUnavailable(PreconditionerName("amg"), "skipped"),
            seed=11,
        )
        == {}
    )


def test_profiles_still_choose_fastest_per_problem_without_reference_selection() -> None:
    from neuralls.domain.benchmarking.costs import solve_seconds
    from neuralls.domain.benchmarking.profiles import performance_profiles

    outcomes = {
        PreconditionerName("slow"): {
            "s0": Solved(MatrixId(1), 1, 3.0, 3.0),
        },
        PreconditionerName("fast"): {
            "s0": Solved(MatrixId(1), 1, 1.0, 1.0),
        },
    }

    profiles = performance_profiles(outcomes, solve_seconds)
    assert profiles[PreconditionerName("fast")].solved_at == (1.0,)
