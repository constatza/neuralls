"""Explicit references never come from observed outcomes."""

from __future__ import annotations

from neuralls.domain.benchmarking.paired import (
    REFERENCE_METRIC,
    reference_among_methods,
    speed_ratio_summary,
)
from neuralls.domain.benchmarking.profiles import performance_series
from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    NoReferenceConfigured,
    ReferenceUnavailable,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.shared.types import PreconditionerName
from tests.observation_support import solved_row, synthetic_observations, unconverged_row


def test_configured_reference_preserves_asymmetric_success_counts() -> None:
    amg, jacobi = PreconditionerName("amg"), PreconditionerName("jacobi")
    observations = synthetic_observations(
        [
            unconverged_row(amg, "s0", 1),
            solved_row(amg, "s1", 1, 4.0),
            solved_row(amg, "s2", 1, 4.0),
            solved_row(jacobi, "s0", 1, 2.0),
            solved_row(jacobi, "s1", 1, 2.0),
            unconverged_row(jacobi, "s2", 1),
        ]
    )

    reference = reference_among_methods(observations, ConfiguredReference(amg))
    assert reference == amg
    row = speed_ratio_summary(observations, amg, REFERENCE_METRIC, seed=11).row(0, named=True)

    assert row["method"] == jacobi
    assert row["n_reference_only"] == 1
    assert row["n_method_only"] == 1
    assert row["n_systems_both_solved"] == 1
    assert row["median_ratio"] == 2.0


def test_no_reference_and_unavailable_reference_have_no_comparison() -> None:
    observations = synthetic_observations(
        [
            solved_row(PreconditionerName("amg"), "s0", 1, 4.0),
            solved_row(PreconditionerName("jacobi"), "s0", 1, 2.0),
        ]
    )

    assert reference_among_methods(observations, NoReferenceConfigured()) is None
    assert (
        reference_among_methods(
            observations, ReferenceUnavailable(PreconditionerName("amg"), "skipped")
        )
        is None
    )


def test_profiles_still_choose_fastest_per_problem_without_reference_selection() -> None:
    slow, fast = PreconditionerName("slow"), PreconditionerName("fast")
    observations = synthetic_observations(
        [solved_row(slow, "s0", 1, 3.0), solved_row(fast, "s0", 1, 1.0)]
    )

    series = performance_series(observations, MetricId.SOLVE_S)

    assert series.frame.filter(series.frame["method"] == fast)["x"].to_list() == [1.0]
    assert series.frame.filter(series.frame["method"] == slow)["x"].to_list() == [3.0]
