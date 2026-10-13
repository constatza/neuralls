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

    outcome = classify_group_aggregation("g1", 2, results_by_system, _SHARING)

    assert isinstance(outcome, GroupAggregated)
    assert outcome.n_aggregated == 2
    assert outcome.n_in_group == 2
    assert set(outcome.projected) == {"jacobi"}


def test_some_members_cache_hits_classifies_as_partially_aggregated() -> None:
    """Fewer real results than group members — still GroupAggregated, with the gap recorded."""
    results_by_system = {"sys-0": {"jacobi": _result("jacobi")}}

    outcome = classify_group_aggregation("g1", 3, results_by_system, _SHARING)

    assert isinstance(outcome, GroupAggregated)
    assert outcome.n_aggregated == 1
    assert outcome.n_in_group == 3


def test_every_member_a_cache_hit_classifies_as_skipped() -> None:
    outcome = classify_group_aggregation("g1", 4, {}, _SHARING)

    assert isinstance(outcome, GroupSkippedAllCacheHit)
    assert outcome.group_id == "g1"
    assert outcome.n_in_group == 4


def test_outcome_cases_are_exhaustively_distinguishable() -> None:
    """Every outcome is exactly one of the two variants — no silent third case."""
    aggregated = classify_group_aggregation(
        "g1", 1, {"sys-0": {"jacobi": _result("jacobi")}}, _SHARING
    )
    skipped = classify_group_aggregation("g1", 1, {}, _SHARING)

    match aggregated:
        case GroupAggregated():
            pass
        case GroupSkippedAllCacheHit():
            raise AssertionError("expected GroupAggregated")

    match skipped:
        case GroupSkippedAllCacheHit():
            pass
        case GroupAggregated():
            raise AssertionError("expected GroupSkippedAllCacheHit")
