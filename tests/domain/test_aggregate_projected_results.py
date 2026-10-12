"""Tests for ``domain.solver.comparison.aggregate_projected_results``.

Pure aggregation over sibling systems' already-produced ``CGComparisonResult``
dicts — never runs a solve itself, so every fixture here is a hand-built
result rather than a real comparison run.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.solver.comparison import aggregate_projected_results
from neuralls.domain.solver.models.result import CGComparisonResult, SetupSharingPolicy, StageCost
from neuralls.shared.types import CostProvenance


def _result(
    name: str,
    *,
    iterations: int,
    setup_cost: float,
    solve_time_seconds: float,
    generation_cost: StageCost | None = None,
) -> CGComparisonResult:
    """Build a minimal converged ``CGComparisonResult`` with explicit costs."""
    return CGComparisonResult(
        x=np.zeros(4),
        converged=True,
        iterations=iterations,
        residual=1e-9,
        residual_abs=1e-9,
        residual_history_rel=[1.0, 1e-9],
        residual_history_abs=[1.0, 1e-9],
        preconditioner=name,
        initial_guess=np.zeros(4),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        generation_cost=generation_cost,
        setup_cost=setup_cost,
        solve_time_seconds=solve_time_seconds,
    )


def test_groups_by_preconditioner_name_across_sibling_systems() -> None:
    """Three sibling systems each reporting "jacobi"/"neural" merge into two projections."""
    results_by_system = {
        "sys-0": {
            "jacobi": _result("jacobi", iterations=5, setup_cost=1.0, solve_time_seconds=2.0),
            "neural": _result("neural", iterations=3, setup_cost=10.0, solve_time_seconds=1.0),
        },
        "sys-1": {
            "jacobi": _result("jacobi", iterations=4, setup_cost=1.1, solve_time_seconds=1.8),
            "neural": _result("neural", iterations=3, setup_cost=10.0, solve_time_seconds=0.9),
        },
    }
    sharing_by_name = {
        "jacobi": SetupSharingPolicy(share_setup=False, share_generation=False),
        "neural": SetupSharingPolicy(share_setup=True, share_generation=False),
    }

    projected = aggregate_projected_results(results_by_system, sharing_by_name)

    assert set(projected) == {"jacobi", "neural"}
    assert projected["jacobi"].n_systems == 2
    assert projected["jacobi"].total_iterations == 9
    assert projected["jacobi"].setup_cost_total_seconds == pytest.approx(2.1)
    assert projected["neural"].setup_cost_total_seconds == pytest.approx(20.0)
    assert projected["neural"].avg_setup_cost_seconds == pytest.approx(10.0)


def test_shared_setup_recovers_the_single_real_value() -> None:
    """When share_setup, every sibling already carries the same real one-time cost."""
    results_by_system = {
        "sys-0": {
            "neural": _result("neural", iterations=3, setup_cost=7.0, solve_time_seconds=1.0)
        },
        "sys-1": {
            "neural": _result("neural", iterations=3, setup_cost=7.0, solve_time_seconds=1.0)
        },
        "sys-2": {
            "neural": _result("neural", iterations=3, setup_cost=7.0, solve_time_seconds=1.0)
        },
    }
    sharing = {"neural": SetupSharingPolicy(share_setup=True, share_generation=False)}

    projected = aggregate_projected_results(results_by_system, sharing)["neural"]

    assert projected.avg_setup_cost_seconds == 7.0
    # Shared: the one-time value appears once, regardless of n.
    assert (
        projected.projected_total_seconds(100)
        == 7.0 + 100 * projected.avg_solve_time_per_system_seconds
    )


def test_unavailable_generation_cost_is_excluded_not_counted_as_zero() -> None:
    """An UNAVAILABLE-provenance generation cost must not silently read as a real zero."""
    unavailable = StageCost(
        wall_time_seconds=0.0, peak_memory_bytes=None, provenance=CostProvenance.UNAVAILABLE
    )
    results_by_system = {
        "sys-0": {
            "pod2g": _result(
                "pod2g",
                iterations=3,
                setup_cost=1.0,
                solve_time_seconds=1.0,
                generation_cost=unavailable,
            )
        },
    }
    sharing = {"pod2g": SetupSharingPolicy(share_setup=True, share_generation=True)}

    projected = aggregate_projected_results(results_by_system, sharing)["pod2g"]

    assert projected.generation_cost_total_seconds is None
    assert projected.avg_generation_cost_seconds is None


def test_historical_generation_cost_sums_across_siblings() -> None:
    historical = StageCost(
        wall_time_seconds=5.0, peak_memory_bytes=None, provenance=CostProvenance.HISTORICAL
    )
    results_by_system = {
        "sys-0": {
            "pod2g": _result(
                "pod2g",
                iterations=3,
                setup_cost=1.0,
                solve_time_seconds=1.0,
                generation_cost=historical,
            )
        },
        "sys-1": {
            "pod2g": _result(
                "pod2g",
                iterations=3,
                setup_cost=1.0,
                solve_time_seconds=1.0,
                generation_cost=historical,
            )
        },
    }
    sharing = {"pod2g": SetupSharingPolicy(share_setup=True, share_generation=True)}

    projected = aggregate_projected_results(results_by_system, sharing)["pod2g"]

    assert projected.generation_cost_total_seconds == pytest.approx(10.0)
    assert projected.avg_generation_cost_seconds == pytest.approx(5.0)


def test_missing_sharing_policy_raises_key_error() -> None:
    results_by_system = {
        "sys-0": {
            "jacobi": _result("jacobi", iterations=5, setup_cost=1.0, solve_time_seconds=2.0)
        },
    }
    with pytest.raises(KeyError):
        aggregate_projected_results(results_by_system, sharing_by_name={})
