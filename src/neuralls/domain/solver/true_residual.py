"""Independent residual measurements and the level at which the solver stops.

Kept apart from ``error_metrics`` because it returns the benchmark-event measurement types,
which ``error_metrics`` (imported by the solver configuration models) must not depend on.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torchalg.strategies.convergence import CombinedToleranceCriterion

from neuralls.domain.solver.models.benchmark_events import (
    MeasuredResidual,
    ResidualMeasurement,
    UnmeasuredResidual,
)


def measure_residual(A: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> ResidualMeasurement:
    """Recompute ``||b - A x||_2 / ||b||_2`` from the returned solution, in ``A``'s dtype.

    This is independent of the solver's recursive residual estimate. A zero right-hand
    side has no relative residual, so it is unmeasured.
    """
    rhs_norm = float(torch.linalg.vector_norm(b))
    if rhs_norm == 0.0:
        return UnmeasuredResidual()
    residual_norm = float(torch.linalg.vector_norm(b - A @ x))
    return MeasuredResidual(rhs_norm=rhs_norm, true_residual_rel=residual_norm / rhs_norm)


def stopping_threshold_rel(rtol: float, atol: float, rhs_norm: float) -> float:
    """Relative residual at which the solver stops: ``max(rtol, atol / ||b||)``.

    Taken from the solver's own criterion so it cannot drift from what the solver applies.
    """
    return float(CombinedToleranceCriterion(rtol=rtol, atol=atol).threshold(rhs_norm)) / rhs_norm


def shared_threshold(thresholds: Sequence[float]) -> float | None:
    """The one threshold every plotted system shares, or ``None`` when they differ."""
    distinct = set(thresholds)
    return distinct.pop() if len(distinct) == 1 else None
