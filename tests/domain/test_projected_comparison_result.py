"""Tests for ``ProjectedComparisonResult``'s scaling-law projection.

Each of setup/generation amortizes or repeats independently per
``SetupSharingPolicy`` — not as one combined "amortized vs recurring"
choice — so these cover every combination explicitly rather than assuming
one implies the other.
"""

from __future__ import annotations

import pytest

from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.sharing import SetupSharingPolicy


def test_fully_amortized_total_is_constant_setup_plus_scaling_solve() -> None:
    """Both setup and generation shared: only the solve term scales with n."""
    result = ProjectedComparisonResult(
        preconditioner_name="neural",
        n_systems=5,
        n_converged=5,
        setup_sharing=SetupSharingPolicy(share_setup=True, share_generation=True),
        setup_cost_total_seconds=10.0,  # same 2.0 repeated 5x
        generation_cost_total_seconds=500.0,  # same 100.0 repeated 5x
        total_solve_time_seconds=25.0,  # 5.0 per system
        total_iterations=50,
    )

    assert result.avg_setup_cost_seconds == 2.0
    assert result.avg_generation_cost_seconds == 100.0
    assert result.avg_solve_time_per_system_seconds == 5.0

    assert result.projected_total_seconds(1) == 2.0 + 100.0 + 5.0
    assert result.projected_total_seconds(100) == 2.0 + 100.0 + 500.0


def test_fully_amortized_avg_per_system_approaches_solve_floor_as_n_grows() -> None:
    """The amortized one-time costs become negligible — avg per system -> avg solve time."""
    result = ProjectedComparisonResult(
        preconditioner_name="neural",
        n_systems=5,
        n_converged=5,
        setup_sharing=SetupSharingPolicy(share_setup=True, share_generation=True),
        setup_cost_total_seconds=50.0,  # 10.0 per system, one-time
        generation_cost_total_seconds=500.0,  # 100.0 per system, one-time
        total_solve_time_seconds=25.0,  # 5.0 per system
        total_iterations=50,
    )

    small_n_avg = result.projected_avg_per_system_seconds(1)
    large_n_avg = result.projected_avg_per_system_seconds(10_000_000)

    assert small_n_avg == 10.0 + 100.0 + 5.0
    assert large_n_avg > result.avg_solve_time_per_system_seconds
    assert abs(large_n_avg - result.avg_solve_time_per_system_seconds) < 1e-3
    assert large_n_avg < small_n_avg


def test_fully_recurring_avg_per_system_never_improves_with_n() -> None:
    """Neither cost amortizes: avg per system is flat, not a plottable improvement."""
    result = ProjectedComparisonResult(
        preconditioner_name="amg",
        n_systems=5,
        n_converged=5,
        setup_sharing=SetupSharingPolicy(share_setup=False, share_generation=False),
        setup_cost_total_seconds=10.0,  # 2.0 per system, repeats
        generation_cost_total_seconds=50.0,  # 10.0 per system, repeats
        total_solve_time_seconds=25.0,  # 5.0 per system
        total_iterations=50,
    )

    per_system_total = 2.0 + 10.0 + 5.0
    assert result.projected_avg_per_system_seconds(1) == per_system_total
    assert result.projected_avg_per_system_seconds(100_000) == per_system_total


def test_setup_and_generation_sharing_are_independent() -> None:
    """Mixed policy: setup shared (fixed-matrix sweep), generation not (inline-fit POD-2G)."""
    result = ProjectedComparisonResult(
        preconditioner_name="pod2g",
        n_systems=4,
        n_converged=4,
        setup_sharing=SetupSharingPolicy(share_setup=True, share_generation=False),
        setup_cost_total_seconds=8.0,  # 2.0 per system, one-time
        generation_cost_total_seconds=40.0,  # 10.0 per system, repeats
        total_solve_time_seconds=8.0,  # 2.0 per system
        total_iterations=40,
    )

    # n=10: setup contributes once (2.0), generation repeats 10x (100.0), solve repeats 10x (20.0).
    assert result.projected_total_seconds(10) == 2.0 + 100.0 + 20.0


def test_generation_not_applicable_contributes_nothing_regardless_of_sharing() -> None:
    """A classical preconditioner with no fit/train step: generation term is always zero."""
    result = ProjectedComparisonResult(
        preconditioner_name="jacobi",
        n_systems=3,
        n_converged=3,
        setup_sharing=SetupSharingPolicy(share_setup=False, share_generation=True),
        setup_cost_total_seconds=3.0,
        generation_cost_total_seconds=None,
        total_solve_time_seconds=9.0,
        total_iterations=30,
    )

    assert result.avg_generation_cost_seconds is None
    assert result.projected_total_seconds(5) == (5 * 1.0) + (5 * 3.0)


def test_avg_iteration_time_sums_across_measured_systems() -> None:
    result = ProjectedComparisonResult(
        preconditioner_name="jacobi",
        n_systems=2,
        n_converged=2,
        setup_sharing=SetupSharingPolicy(share_setup=False, share_generation=False),
        setup_cost_total_seconds=1.0,
        generation_cost_total_seconds=None,
        total_solve_time_seconds=10.0,
        total_iterations=20,
    )

    assert result.avg_iteration_time_seconds == 0.5


def test_partial_success_divides_repeated_terms_by_success_rate_and_keeps_shared_once() -> None:
    """5 systems, 3 solved: shared terms stay one-time, repeated ones scale by 1 / 0.6."""

    def build(sharing: SetupSharingPolicy) -> ProjectedComparisonResult:
        return ProjectedComparisonResult(
            preconditioner_name="x",
            n_systems=5,
            n_converged=3,
            setup_sharing=sharing,
            setup_cost_total_seconds=10.0,  # 2.0 per system
            generation_cost_total_seconds=50.0,  # 10.0 per system
            total_solve_time_seconds=6.2,  # 1.24 per system over all runs
            total_iterations=230,
        )

    shared = build(SetupSharingPolicy(share_setup=True, share_generation=True))
    repeated = build(SetupSharingPolicy(share_setup=False, share_generation=False))

    assert shared.success_rate == pytest.approx(0.6)
    assert shared.avg_solve_time_per_solved_system_seconds == pytest.approx(6.2 / 3)
    assert shared.one_time_seconds == pytest.approx(12.0)
    assert shared.per_solved_system_seconds == pytest.approx(6.2 / 3)
    assert shared.projected_total_seconds(10) == pytest.approx(12.0 + 10 * 6.2 / 3)
    assert repeated.one_time_seconds == 0.0
    assert repeated.per_solved_system_seconds == pytest.approx((2.0 + 10.0 + 1.24) / 0.6)
    assert repeated.projected_total_seconds(10) == pytest.approx(10 * 13.24 / 0.6)


def test_avg_iteration_time_is_total_solve_over_total_iterations() -> None:
    result = ProjectedComparisonResult(
        preconditioner_name="x",
        n_systems=5,
        n_converged=3,
        setup_sharing=SetupSharingPolicy(share_setup=False, share_generation=False),
        setup_cost_total_seconds=0.0,
        generation_cost_total_seconds=None,
        total_solve_time_seconds=6.2,
        total_iterations=230,
    )

    assert result.avg_iteration_time_seconds == pytest.approx(6.2 / 230)


def test_full_success_matches_plain_per_system_averages() -> None:
    result = ProjectedComparisonResult(
        preconditioner_name="x",
        n_systems=4,
        n_converged=4,
        setup_sharing=SetupSharingPolicy(share_setup=True, share_generation=False),
        setup_cost_total_seconds=8.0,
        generation_cost_total_seconds=40.0,
        total_solve_time_seconds=8.0,
        total_iterations=40,
    )

    assert result.success_rate == 1.0
    assert result.per_solved_system_seconds == pytest.approx(10.0 + 2.0)
    assert (
        result.avg_solve_time_per_solved_system_seconds == result.avg_solve_time_per_system_seconds
    )


def test_avg_iteration_time_none_when_no_iterations_measured() -> None:
    result = ProjectedComparisonResult(
        preconditioner_name="jacobi",
        n_systems=1,
        n_converged=1,
        setup_sharing=SetupSharingPolicy(share_setup=False, share_generation=False),
        setup_cost_total_seconds=1.0,
        generation_cost_total_seconds=None,
        total_solve_time_seconds=1.0,
        total_iterations=0,
    )

    assert result.avg_iteration_time_seconds is None
