"""Ranked recommendations from a comparison run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankedRecommendation:
    """A single ranked entry from a solver comparison.

    Args:
        method_id: Id of the ranked method; its description comes from its presentation.
        iterations: Iterations to converge.
        residual: Final relative residual.
        residual_abs: Final absolute residual.
        breakdown: Whether breakdown occurred.
        solve_time_seconds: Diagnostic PCG/FCG execution time, the ranking key;
            `None` ranks after every measured solve.
    """

    method_id: str
    iterations: int
    residual: float
    residual_abs: float
    breakdown: bool
    solve_time_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ComparisonRecommendations:
    """Ranked recommendations from a comparison run.

    Args:
        ranked: All converged entries sorted by solve time (best first).
        overall_best: The single best entry, or None if none converged.
    """

    ranked: tuple[RankedRecommendation, ...] = ()
    overall_best: RankedRecommendation | None = None
