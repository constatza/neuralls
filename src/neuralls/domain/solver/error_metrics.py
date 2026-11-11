"""Reference-solution and energy-norm error metrics for CG comparisons.

Pure numeric helpers — reference factorizations and solves run on the host,
no exception-swallowing. Callers (``comparison.py``) own the resilience *policy*
(catching failures, falling back to ``None`` on an unset metric); this module only
defines what the numbers mean, returns ``None``/raises on well-defined edge cases,
and warns (but does not fail) when a requested precision is silently downgraded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.sparse import csr_array
from scipy.sparse.linalg import splu
from torchalg.monitoring.analysis import golub_meurant_error_bound

if TYPE_CHECKING:
    from torchalg.models.result import SolverResult

REFERENCE_PRECISION_MARGIN = 1e-3
"""Default reference residual target is three orders below the comparison rtol."""

FLOAT64_EPS = torch.finfo(torch.float64).eps
"""Machine epsilon for float64 (~2.22e-16)."""

MIN_SAFE_RELATIVE_RESIDUAL = 10 * FLOAT64_EPS
"""Floor for the reference solve's target relative residual, with headroom above the
float64 noise floor so the solve converges to a real value rather than noise."""

GOLUB_MEURANT_DELAY = 10


def _reference_target(rtol: float, margin: float) -> float:
    """Relative residual a reference solution must reach for a comparison at ``rtol``."""
    if margin >= 1:
        raise ValueError(
            f"reference_precision_margin must be < 1 (got {margin!r}); a margin >= 1 "
            "would make the reference no more precise than the solvers being compared."
        )
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
    return target_rel


@dataclass(frozen=True, slots=True)
class ReferenceSolver:
    """One direct factorization of a system matrix, reused for each of its right-hand sides.

    Serves only the reference solution ``x*`` that exact-error metrics compare against;
    preconditioners and the compared solvers never see it. Factoring once per matrix
    keeps sibling right-hand sides from repeating the factorization.
    """

    matrix: torch.Tensor
    """The factored matrix in float64, kept for the true-residual check."""
    host_solve: Callable[[np.ndarray], np.ndarray]
    """Solves with the host factorization for one float64 right-hand side."""

    def solve(
        self, b: torch.Tensor, *, rtol: float, margin: float = REFERENCE_PRECISION_MARGIN
    ) -> torch.Tensor:
        """Reference solution of ``A x = b`` with a checked true residual.

        The accepted residual is
        ``||b - A x|| / ||b|| <= max(rtol * margin, MIN_SAFE_RELATIVE_RESIDUAL)``, so the
        default is three orders tighter than the compared solver, capped above the
        float64 noise floor.

        Args:
            b (torch.Tensor): Right-hand side, shape ``(n,)``.
            rtol (float): Relative tolerance requested from the compared solvers.
            margin (float): How many orders of magnitude tighter than ``rtol`` the
                reference must be (default ``1e-3``). Must be ``< 1`` — a reference no
                more precise than the solvers being compared is useless.

        Returns:
            torch.Tensor: Reference solution ``x*`` in float64, on the matrix's device.

        Raises:
            ValueError: If ``margin >= 1``.
            RuntimeError: If the true residual is not finite or exceeds the target.
        """
        target_rel = _reference_target(rtol, margin)
        b = b.detach().double().to(self.matrix.device)
        host_solution = self.host_solve(b.cpu().numpy())
        x = torch.as_tensor(np.asarray(host_solution), dtype=torch.float64, device=b.device)
        rhs_norm = float(torch.linalg.vector_norm(b))
        residual_rel = (
            float(torch.linalg.vector_norm(b - self.matrix @ x)) / rhs_norm if rhs_norm else 0.0
        )
        if not np.isfinite(residual_rel) or residual_rel > target_rel:
            raise RuntimeError(
                f"Reference direct solve could not reach relative residual {target_rel:.1e}; "
                f"achieved {residual_rel:.1e}."
            )
        return x


def reference_solver(A: torch.Tensor) -> ReferenceSolver:
    """Factor ``A`` once on the host for reference solves.

    CSR matrices use SciPy's sparse LU (``splu``) without densifying; dense matrices use
    a Cholesky factorization. Both run on the CPU, outside any timed region.

    Args:
        A (torch.Tensor): System matrix, shape ``(n, n)``, strided or ``torch.sparse_csr``.

    Raises:
        ValueError: If ``A`` has another layout.
        RuntimeError: If the factorization fails (singular, or not positive definite).
    """
    A = A.detach().double()
    try:
        if A.layout is torch.sparse_csr:
            host_matrix = csr_array(
                (
                    A.values().cpu().numpy(),
                    A.col_indices().cpu().numpy(),
                    A.crow_indices().cpu().numpy(),
                ),
                shape=(A.shape[0], A.shape[1]),
            )
            return ReferenceSolver(A, splu(host_matrix.tocsc()).solve)
        if A.layout is torch.strided:
            return ReferenceSolver(A, partial(cho_solve, cho_factor(A.cpu().numpy())))
    except (LinAlgError, RuntimeError) as exc:
        raise RuntimeError(f"Reference factorization failed: {exc}") from exc
    raise ValueError(f"Unsupported reference matrix layout: {A.layout}")


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
