"""Performance and ECDF profiles (``domain.benchmarking.profiles``): failures stay unsolved."""

from __future__ import annotations

import pytest

from neuralls.domain.benchmarking.costs import iterations, seconds
from neuralls.domain.benchmarking.profiles import ecdf_profiles, performance_profiles
from neuralls.domain.benchmarking.selection import reference_preconditioner, systems_solved_by_all
from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome


@pytest.fixture
def outcomes() -> dict[str, dict[str, SystemOutcome]]:
    """Four systems; ``amg`` solves all, ``jacobi`` fails s2 and s3."""
    return {
        "amg": {
            "s0": Solved(iterations=5, recurring_time_s=1.0),
            "s1": Solved(iterations=6, recurring_time_s=2.0),
            "s2": Solved(iterations=7, recurring_time_s=4.0),
            "s3": Solved(iterations=8, recurring_time_s=8.0),
        },
        "jacobi": {
            "s0": Solved(iterations=10, recurring_time_s=2.0),
            "s1": Solved(iterations=12, recurring_time_s=2.0),
            "s2": Failed(),
            "s3": Failed(),
        },
    }


def test_performance_profile_scores_failures_as_unsolved(outcomes) -> None:
    profiles = performance_profiles(outcomes, seconds)

    assert profiles["amg"].solved_at == (1.0, 1.0, 1.0, 1.0)
    assert profiles["jacobi"].solved_at == (1.0, 2.0)
    assert profiles["jacobi"].fraction_solved(2.0) == pytest.approx(0.5)
    assert profiles["jacobi"].fraction_solved(1e9) == pytest.approx(0.5)


def test_systems_no_method_solved_are_excluded_from_every_performance_profile() -> None:
    outcomes = {
        "a": {"s0": Solved(1, 1.0), "s1": Failed()},
        "b": {"s0": Solved(1, 2.0), "s1": Failed()},
    }

    profiles = performance_profiles(outcomes, seconds)

    assert profiles["a"].n_systems == profiles["b"].n_systems == 1


def test_ecdf_profile_is_absolute_and_counts_failures_in_the_denominator(outcomes) -> None:
    profiles = ecdf_profiles(outcomes, iterations)

    assert profiles["jacobi"].n_systems == 4
    assert profiles["jacobi"].fraction_solved(11.0) == pytest.approx(0.25)
    assert profiles["amg"].fraction_solved(6.0) == pytest.approx(0.5)


def test_solved_run_with_incomparable_time_is_dropped_from_the_time_profile() -> None:
    outcomes = {"a": {"s0": Solved(1, None), "s1": Solved(1, 1.0)}}

    assert ecdf_profiles(outcomes, seconds)["a"].n_systems == 1


def test_incomparable_time_in_one_method_removes_the_system_for_every_method() -> None:
    outcomes = {
        "a": {"s0": Solved(1, None), "s1": Solved(1, 1.0), "s2": Solved(1, 1.0)},
        "b": {"s0": Solved(1, 2.0), "s1": Failed(), "s2": Solved(1, 3.0)},
    }

    ecdf = ecdf_profiles(outcomes, seconds)
    perf = performance_profiles(outcomes, seconds)

    assert ecdf["a"].n_systems == ecdf["b"].n_systems == 2
    assert ecdf["a"].solved_at == (1.0, 1.0)
    assert ecdf["b"].solved_at == (3.0,)
    assert perf["a"].n_systems == perf["b"].n_systems == 2


def test_zero_cost_run_is_excluded_for_every_method() -> None:
    outcomes = {
        "a": {"s0": Solved(1, 0.0), "s1": Solved(1, 1.0)},
        "b": {"s0": Solved(1, 5.0), "s1": Solved(1, 2.0)},
    }

    profiles = ecdf_profiles(outcomes, seconds)

    assert profiles["a"].n_systems == profiles["b"].n_systems == 1
    assert profiles["b"].solved_at == (2.0,)


def test_plateau_is_the_success_rate_over_the_common_systems(outcomes) -> None:
    profiles = ecdf_profiles(outcomes, seconds)

    assert profiles["jacobi"].fraction_solved(1e9) == pytest.approx(0.5)
    assert profiles["amg"].fraction_solved(1e9) == pytest.approx(1.0)


def test_reference_tie_goes_to_the_lowest_total_time_then_plan_order() -> None:
    slow = {"s0": Solved(1, 5.0), "s1": Failed()}
    fast = {"s0": Solved(1, 1.0), "s1": Failed()}

    assert reference_preconditioner({"slow": slow, "fast": fast}) == "fast"
    assert reference_preconditioner({"first": fast, "second": dict(fast)}) == "first"


def test_reference_is_the_method_that_solved_the_most_systems(outcomes) -> None:
    assert reference_preconditioner(outcomes) == "amg"


def test_reference_of_an_empty_group_is_none() -> None:
    assert reference_preconditioner({}) is None


def test_systems_solved_by_all_is_the_intersection_of_solved_sets(outcomes) -> None:
    assert systems_solved_by_all(outcomes) == {"s0", "s1"}


def test_converged_at_the_initial_guess_keeps_its_system_in_the_iteration_ecdf() -> None:
    outcomes = {"a": {"s0": Solved(0, 1.0), "s1": Solved(5, 1.0)}}

    profile = ecdf_profiles(outcomes, iterations)["a"]

    assert profile.n_systems == 2
    assert profile.fraction_solved(0.0) == pytest.approx(0.5)
