"""Tests for ``domain.solver.comparison.classify_group_aggregation``.

Pure classification: given already-filtered real results (cache hits
dropped) plus the group's total member count, decide which of the two
``ComparisonGroupOutcome`` cases applies. No MLflow or I/O involved.
"""

from __future__ import annotations

import numpy as np

from neuralls.domain.solver.comparison import classify_group_aggregation
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    GroupAggregated,
    GroupBelowMinimumSize,
    GroupSkippedAllCacheHit,
    SetupSharingPolicy,
)


def _result(name: str) -> CGComparisonResult:
    return CGComparisonResult(
        x=np.zeros(2),
        converged=True,
        iterations=3,
        residual=1e-9,
        residual_abs=1e-9,
        residual_history_rel=[1.0, 1e-9],
        residual_history_abs=[1.0, 1e-9],
        preconditioner=name,
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        setup_cost=1.0,
        solve_time_seconds=0.5,
    )


_SHARING = {"jacobi": SetupSharingPolicy(share_setup=False, share_generation=False)}


def test_all_members_contributing_classifies_as_fully_aggregated() -> None:
    results_by_system = {
        "sys-0": {"jacobi": _result("jacobi")},
        "sys-1": {"jacobi": _result("jacobi")},
    }

    outcome = classify_group_aggregation("g1", 2, results_by_system, _SHARING, min_size=1)

    assert isinstance(outcome, GroupAggregated)
    assert outcome.n_aggregated == 2
    assert outcome.n_in_group == 2
    assert set(outcome.projected) == {"jacobi"}


def test_some_members_cache_hits_classifies_as_partially_aggregated() -> None:
    """Fewer real results than group members — still GroupAggregated, with the gap recorded."""
    results_by_system = {"sys-0": {"jacobi": _result("jacobi")}}

    outcome = classify_group_aggregation("g1", 3, results_by_system, _SHARING, min_size=1)

    assert isinstance(outcome, GroupAggregated)
    assert outcome.n_aggregated == 1
    assert outcome.n_in_group == 3


def test_every_member_a_cache_hit_classifies_as_skipped() -> None:
    outcome = classify_group_aggregation("g1", 4, {}, _SHARING, min_size=1)

    assert isinstance(outcome, GroupSkippedAllCacheHit)
    assert outcome.group_id == "g1"
    assert outcome.n_in_group == 4


def test_fewer_contributors_than_the_minimum_classifies_as_below_minimum_size() -> None:
    results_by_system = {"sys-0": {"jacobi": _result("jacobi")}}

    outcome = classify_group_aggregation("g1", 4, results_by_system, _SHARING, min_size=3)

    assert isinstance(outcome, GroupBelowMinimumSize)
    assert (outcome.n_aggregated, outcome.n_in_group, outcome.min_size) == (1, 4, 3)


def test_exactly_the_minimum_contributors_is_aggregated() -> None:
    results_by_system = {f"sys-{i}": {"jacobi": _result("jacobi")} for i in range(3)}

    outcome = classify_group_aggregation("g1", 3, results_by_system, _SHARING, min_size=3)

    assert isinstance(outcome, GroupAggregated)


def test_nothing_measured_is_skipped_even_when_the_minimum_is_unmet() -> None:
    """All-cache-hit is its own case, not "too small" — it needs a different message."""
    outcome = classify_group_aggregation("g1", 4, {}, _SHARING, min_size=3)

    assert isinstance(outcome, GroupSkippedAllCacheHit)


def test_outcome_cases_are_exhaustively_distinguishable() -> None:
    """Every outcome is exactly one of the three variants — match must cover all."""
    one = {"sys-0": {"jacobi": _result("jacobi")}}
    outcomes = [
        classify_group_aggregation("g1", 1, one, _SHARING, min_size=1),
        classify_group_aggregation("g1", 1, {}, _SHARING, min_size=1),
        classify_group_aggregation("g1", 1, one, _SHARING, min_size=2),
    ]

    labels = []
    for outcome in outcomes:
        match outcome:
            case GroupAggregated():
                labels.append("aggregated")
            case GroupSkippedAllCacheHit():
                labels.append("skipped")
            case GroupBelowMinimumSize():
                labels.append("too_small")

    assert labels == ["aggregated", "skipped", "too_small"]
