"""Solve one system the way a comparison run does, for solver-level tests."""

from __future__ import annotations

import torch
from torchalg.preconditioners.base import Preconditioner

from neuralls.domain.solver.comparison import (
    compute_reference_solution,
    prepare_reference,
    solve_system,
)
from neuralls.domain.solver.error_metrics import reference_target
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.config import REFERENCE_PRECISION_MARGIN, SolverParams
from neuralls.shared.constants import DEFAULT_ATOL, DEFAULT_M_MAX

DEFAULT_TEST_MAXITER = 100


def solver_params(rtol: float = 1e-8, maxiter: int = DEFAULT_TEST_MAXITER) -> SolverParams:
    """Comparison solver parameters with the defaults a run uses."""
    return SolverParams(
        rtol=rtol,
        atol=DEFAULT_ATOL,
        max_iterations=maxiter,
        stopping_criterion="residual_norm",
        m_max=DEFAULT_M_MAX,
    )


def solve_against_reference(
    A: torch.Tensor,
    b: torch.Tensor,
    preconditioner: Preconditioner,
    *,
    rtol: float = 1e-8,
    maxiter: int = DEFAULT_TEST_MAXITER,
) -> CGComparisonResult:
    """Factor ``A``, compute this system's reference solution, then solve it once."""
    target_rel = reference_target(rtol, REFERENCE_PRECISION_MARGIN).applied_rel
    reference = compute_reference_solution(prepare_reference(A), b, target_rel=target_rel)
    return solve_system(
        A,
        b,
        preconditioner,
        name="method",
        reference=reference,
        params=solver_params(rtol, maxiter),
    )
