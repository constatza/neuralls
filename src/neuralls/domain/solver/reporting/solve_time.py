"""Time to solution of one run."""

from __future__ import annotations

from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.shared.types import CostProvenance


def available_generation_seconds(result: CGComparisonResult) -> float | None:
    cost = result.generation_cost
    if cost is None or cost.provenance is CostProvenance.UNAVAILABLE:
        return None
    return cost.wall_time_seconds


def time_to_solution(result: CGComparisonResult) -> float | None:
    """Total time, or `None` when an unavailable cost makes it incomparable."""
    return None if result.has_unavailable_cost else result.total_time_seconds
