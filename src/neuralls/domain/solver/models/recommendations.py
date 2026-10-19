"""Ranked recommendations from a comparison run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankedRecommendation:
    """A single ranked entry from a solver comparison.

    Args:
        label: Preconditioner/solver label.
        iterations: Iterations to converge.
        residual: Final relative residual.
        residual_abs: Final absolute residual.
        breakdown: Whether breakdown occurred.
        total_time_seconds: Time-to-solution (generation + setup + solve), the
            ranking key; `None` when it cannot be compared fairly (an
            unavailable-provenance cost), which ranks after every timed entry.
    """

    label: str
    iterations: int
    residual: float
    residual_abs: float
    breakdown: bool
    total_time_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ComparisonRecommendations:
    """Ranked recommendations from a comparison run.

    Args:
        ranked: All converged entries sorted by time-to-solution (best first).
        overall_best: The single best entry, or None if none converged.
    """

    ranked: tuple[RankedRecommendation, ...] = ()
    overall_best: RankedRecommendation | None = None
