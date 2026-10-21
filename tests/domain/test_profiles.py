"""Failure-aware profiles over explicit solve and amortized-online costs."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.domain.benchmarking.costs import (
    amortized_online_seconds,
    iterations,
    solve_seconds,
)
from neuralls.domain.benchmarking.profiles import ecdf_profiles, performance_profiles
from neuralls.domain.benchmarking.selection import systems_solved_by_all
from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome
from neuralls.shared.types import MatrixId


@pytest.fixture
def outcomes() -> dict[str, dict[str, SystemOutcome]]:
    """Four systems; ``amg`` solves all, ``jacobi`` fails two without numeric costs."""
    return {
        "amg": {
            "s0": Solved(MatrixId(10), iterations=5, solve_s=1.0, amortized_online_s=1.5),
            "s1": Solved(MatrixId(10), iterations=6, solve_s=2.0, amortized_online_s=2.5),
            "s2": Solved(MatrixId(20), iterations=7, solve_s=4.0, amortized_online_s=4.5),
            "s3": Solved(MatrixId(20), iterations=8, solve_s=8.0, amortized_online_s=8.5),
        },
        "jacobi": {
            "s0": Solved(MatrixId(10), iterations=10, solve_s=2.0, amortized_online_s=3.0),
            "s1": Solved(MatrixId(10), iterations=12, solve_s=2.0, amortized_online_s=3.0),
            "s2": Failed(MatrixId(20), iterations=100, reason="iteration limit"),
            "s3": Failed(MatrixId(20), iterations=4, reason="breakdown"),
        },
    }


@pytest.fixture(params=(solve_seconds, amortized_online_seconds))
def time_cost(request: pytest.FixtureRequest) -> Callable[[Solved], float]:
    """Every time-profile contract is exercised for both supported scenarios."""
    return request.param


def test_performance_profile_scores_failures_as_unsolved(outcomes, time_cost) -> None:
    profiles = performance_profiles(outcomes, time_cost)

    assert profiles["amg"].solved_at == (1.0, 1.0, 1.0, 1.0)
    expected = (1.0, 2.0) if time_cost is solve_seconds else (1.2, 2.0)
    assert profiles["jacobi"].solved_at == pytest.approx(expected)
    assert profiles["jacobi"].fraction_solved(1e9) == pytest.approx(0.5)


def test_systems_no_method_solved_remain_in_performance_profile_denominator(time_cost) -> None:
    outcomes = {
        "a": {
            "s0": Solved(MatrixId(1), 1, 1.0, 1.5),
            "s1": Failed(MatrixId(2), 3, "limit"),
        },
        "b": {
            "s0": Solved(MatrixId(1), 1, 2.0, 2.5),
            "s1": Failed(MatrixId(2), 3, "limit"),
        },
    }

    profiles = performance_profiles(outcomes, time_cost)

    assert profiles["a"].n_systems == profiles["b"].n_systems == 2
    assert profiles["a"].fraction_solved(1e9) == pytest.approx(0.5)


def test_all_failed_methods_still_have_zero_profile_plateaus(time_cost) -> None:
    outcomes = {
        "a": {"s0": Failed(MatrixId(1), 3, "setup failed")},
        "b": {"s0": Failed(MatrixId(1), 3, "interrupted")},
    }

    profiles = performance_profiles(outcomes, time_cost)

    assert set(profiles) == {"a", "b"}
    assert all(profile.n_systems == 1 and not profile.solved_at for profile in profiles.values())


def test_ecdf_counts_failures_in_the_denominator(outcomes, time_cost) -> None:
    profiles = ecdf_profiles(outcomes, time_cost)

    assert profiles["jacobi"].n_systems == 4
    assert profiles["jacobi"].fraction_solved(1e9) == pytest.approx(0.5)
    assert profiles["amg"].fraction_solved(1e9) == pytest.approx(1.0)


def test_failed_observation_has_no_numeric_cost_fields(outcomes, time_cost) -> None:
    failure = outcomes["jacobi"]["s2"]

    assert isinstance(failure, Failed)
    assert vars(type(failure))["__slots__"] == ("matrix_id", "iterations", "reason")
    assert not hasattr(failure, "solve_s")
    assert not hasattr(failure, "amortized_online_s")


def test_fastest_method_is_selected_per_problem_not_globally(outcomes, time_cost) -> None:
    profiles = performance_profiles(outcomes, time_cost)

    assert profiles["amg"].fraction_solved(1.0) == pytest.approx(1.0)
    assert profiles["jacobi"].fraction_solved(1.0) == pytest.approx(
        0.25 if time_cost is solve_seconds else 0.0
    )


def test_systems_solved_by_all_is_the_intersection_of_solved_sets(outcomes) -> None:
    assert systems_solved_by_all(outcomes) == {"s0", "s1"}


def test_initial_guess_convergence_remains_in_iteration_ecdf() -> None:
    outcomes = {
        "a": {
            "s0": Solved(MatrixId(1), 0, 0.0, 0.5),
            "s1": Solved(MatrixId(1), 5, 1.0, 1.5),
        }
    }

    profile = ecdf_profiles(outcomes, iterations)["a"]

    assert profile.n_systems == 2
    assert profile.fraction_solved(0.0) == pytest.approx(0.5)
