"""Reference-solution and energy-norm error metrics for CG comparisons.

Pure numeric helpers — no device transfers beyond what a single computation needs,
no exception-swallowing. Callers (``comparison.py``) own the resilience *policy*
(catching failures, falling back to ``None`` on an unset metric); this module only
defines what the numbers mean, returns ``None``/raises on well-defined edge cases,
and warns (but does not fail) when a requested precision is silently downgraded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from loguru import logger
from torchalg import pcg
from torchalg.monitoring import TraceMode
from torchalg.monitoring.analysis import golub_meurant_error_bound
from torchalg.preconditioners.implementations import JacobiPreconditioner

if TYPE_CHECKING:
    from torchalg.models.result import SolverResult

REFERENCE_PRECISION_MARGIN = 1e-2
REFERENCE_MAX_ITER_PER_DIM = 50
"""Jacobi-PCG iteration budget for the reference solve, as a multiple of system size.
Generous since each iteration is O(n) (matvec-dominated) rather than the O(n^3) a dense
direct solve would cost, and the reference target is far tighter than a normal rtol."""

FLOAT64_EPS = torch.finfo(torch.float64).eps
"""Machine epsilon for float64 (~2.22e-16)."""

MIN_SAFE_RELATIVE_RESIDUAL = 10 * FLOAT64_EPS
"""Floor for the reference solve's target relative residual, with headroom above the
float64 noise floor so the solve converges to a real value rather than noise."""

GOLUB_MEURANT_DELAY = 10


def reference_solution(
    A: torch.Tensor, b: torch.Tensor, *, rtol: float, margin: float = REFERENCE_PRECISION_MARGIN
) -> torch.Tensor:
    """Jacobi-PCG solve of ``A x = b``, accurate well beyond the comparison tolerance.

    Uses the same ``torchalg.pcg`` engine the comparisons themselves run,
    Jacobi-preconditioned and driven to a far tighter tolerance, rather than a
    dense direct solve: a direct solve is ``O(n^3)`` per factorization, which
    stops scaling long before the systems this module is comparing
    preconditioners on do; Jacobi-PCG stays ``O(n)`` per iteration regardless
    of system size. Runs until
    ``||b - A x|| / ||b|| <= max(rtol * margin, MIN_SAFE_RELATIVE_RESIDUAL)``, so the
    reference error is orders of magnitude below what the solvers under comparison
    are asked to reach, without ever asking for a target below float64 machine
    precision (which no solver could reach, or would reach only as noise).

    Args:
        A (torch.Tensor): System matrix, shape ``(n, n)``.
        b (torch.Tensor): Right-hand side, shape ``(n,)``.
        rtol (float): Relative tolerance requested from the compared solvers.
        margin (float): How many orders of magnitude tighter than ``rtol`` the
            reference must be (default ``1e-2``). Must be ``< 1`` — a reference no
            more precise than the solvers being compared is useless.

    Returns:
        torch.Tensor: Reference solution ``x*`` in float64 (reaching the
        target precision is impossible in lower precision).

    Raises:
        ValueError: If ``margin >= 1``.
        RuntimeError: If Jacobi-PCG cannot reach the target precision within
            its iteration budget.
    """
    if margin >= 1:
        raise ValueError(
            f"reference_precision_margin must be < 1 (got {margin!r}); a margin >= 1 "
            "would make the reference no more precise than the solvers being compared."
        )
    A, b = A.double(), b.double()
    requested_rel = rtol * margin
    target_rel = max(requested_rel, MIN_SAFE_RELATIVE_RESIDUAL)
    if target_rel > requested_rel:
        logger.warning(
            "Requested reference precision margin {:.1e} would ask below float64 machine "
            "precision at rtol={:.1e}; clamped to {:.1e} relative residual.",
            margin,
            rtol,
            target_rel,
        )
    maxiter = REFERENCE_MAX_ITER_PER_DIM * b.numel()
    x, info = pcg(
        A,
        b,
        torch.zeros_like(b),
        preconditioner=JacobiPreconditioner(A),
        rtol=target_rel,
        atol=0.0,
        maxiter=maxiter,
        trace_mode=TraceMode.DISABLED,
    )
    if not info.converged:
        raise RuntimeError(
            f"Reference solution could not reach relative residual {target_rel:.1e} within "
            f"{maxiter} Jacobi-PCG iterations; the system is too ill-conditioned at this rtol."
        )
    return x


def relative_exact_error(x_sol: torch.Tensor, x_exact: torch.Tensor) -> float:
    """Relative error ``||x_sol - x_exact|| / ||x_exact||``, or absolute if ``x_exact`` is ~0.

    Aligns ``x_exact`` to ``x_sol``'s device before combining them — device only,
    never dtype, so a real precision mismatch stays visible instead of being
    silently downcast.

    Args:
        x_sol (torch.Tensor): Solver's returned solution.
        x_exact (torch.Tensor): Reference solution.

    Returns:
        float: Relative (or absolute, if ``x_exact`` is zero) error.
    """
    x_exact_matched = x_exact.to(device=x_sol.device)
    exact_norm = float(torch.linalg.vector_norm(x_exact_matched))
    diff_norm = float(torch.linalg.vector_norm(x_sol - x_exact_matched))
    if exact_norm == 0:
        return diff_norm
    return diff_norm / exact_norm


def normalize_error_history(history: Sequence[float] | None) -> list[float] | None:
    """Normalize an error/bound history by its first entry so every curve starts at 1.

    Args:
        history (Sequence[float] | None): Per-iteration error or error-bound values.

    Returns:
        list[float] | None: ``history`` divided by ``history[0]``, or ``None`` if
        ``history`` is empty or its first entry is zero.
    """
    if not history:
        return None
    first = history[0]
    if first == 0:
        return None
    return [value / first for value in history]


def extract_energy_error(
    result: SolverResult,
) -> tuple[list[float] | None, list[float] | None]:
    """Relative energy-norm error, exact if available, else a Golub-Meurant lower bound.

    ``||v||_A = sqrt(v^T A v)`` is the norm CG minimizes, so this is the conventional
    convergence measure for (P)CG. The exact value (``result.error_history_a_norm``)
    is only populated when ``x_exact`` was passed into the solve; when it isn't (e.g.
    the reference solve failed), ``result.energy_decrements`` — always available —
    feeds a ground-truth-free lower bound on the same quantity instead. Only one of
    the two return slots is ever non-``None``: a bound must never masquerade as the
    exact error.

    Args:
        result (SolverResult): Solver diagnostics returned by ``pcg``/``flexible_cg``.

    Returns:
        tuple[list[float] | None, list[float] | None]: ``(error_history_a_rel,
        error_bound_a_rel)``.
    """
    if result.error_history_a_norm:
        return normalize_error_history(result.error_history_a_norm), None
    if result.energy_decrements:
        bound = golub_meurant_error_bound(result.energy_decrements, delay=GOLUB_MEURANT_DELAY)
        return None, normalize_error_history(bound)
    return None, None
