"""Minimal solver results carrying convergence histories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neuralls.domain.solver.models.cg_result import CGComparisonResult


def cg_result(
    residuals: Sequence[float],
    energy: Sequence[float] | None = None,
    *,
    converged: bool = True,
) -> CGComparisonResult:
    """A solver result whose only meaningful content is its per-iteration histories."""
    return CGComparisonResult(
        x=np.zeros(2),
        converged=converged,
        iterations=len(residuals),
        residual=residuals[-1],
        residual_abs=residuals[-1],
        residual_history_rel=list(residuals),
        residual_history_abs=list(residuals),
        preconditioner="p",
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        error_history_a_rel=list(energy) if energy is not None else None,
    )
