"""Time to solution of one run, and its recurring part under a sharing policy."""

from __future__ import annotations

from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
from neuralls.shared.types import CostProvenance


def available_generation_seconds(result: CGComparisonResult) -> float | None:
    cost = result.generation_cost
    if cost is None or cost.provenance is CostProvenance.UNAVAILABLE:
        return None
    return cost.wall_time_seconds


def recurring_seconds(result: CGComparisonResult, policy: SetupSharingPolicy) -> float | None:
    """Time to solution minus the setup and generation the policy flags as one-time.

    `None` exactly when `time_to_solution` is. A shared cost is a copy of one
    real value the caching layer wrote into every sibling, so it is charged
    once (in ERT, the projection and break-even), not in each system's time.
    """
    total = time_to_solution(result)
    if total is None:
        return None
    return total - policy.one_time_seconds(result.setup_cost, available_generation_seconds(result))


def time_to_solution(result: CGComparisonResult) -> float | None:
    """Total time, or `None` when it is incomparable.

    Incomparable means an unavailable cost, or no measured solve at all (a
    preconditioner that failed to build): a zero there would read as free.
    """
    if result.has_unavailable_cost or result.solve_time_seconds is None:
        return None
    return result.total_time_seconds
