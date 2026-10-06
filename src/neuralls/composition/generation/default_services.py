"""Default collaborators for generation workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

from neuralls.composition.solvers import run_traced_pcg
from neuralls.domain.generation.ports import TracingSolverPort


def make_solver() -> TracingSolverPort:
    """Construct the default tracing solver for single-RHS trace strategies.

    Used by the residual (``residuals``/``gaussian_residuals``) strategies,
    which trace the torchalg PCG call with full iteration history.
    """

    def _solve(
        A: np.ndarray,
        b: np.ndarray,
        x0: np.ndarray,
        *,
        maxiter: int,
        rtol: float,
        atol: float,
    ) -> tuple[np.ndarray, Any]:
        return run_traced_pcg(A, b, x0, maxiter=maxiter, rtol=rtol, atol=atol)

    return _solve
