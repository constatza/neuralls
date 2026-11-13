"""Reference-solution and energy-norm error metrics for CG comparisons.

Pure numeric helpers — reference factorizations and solves run on the host,
no exception-swallowing. Callers (``comparison.py``) own the resilience *policy*
(turning a failed metric into its unavailable variant); this module only defines what
the numbers mean and raises on well-defined edge cases. It never logs: a clamped
reference precision is returned as ``ReferenceTarget`` for the caller to report.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.sparse import csr_array
from scipy.sparse.linalg import splu
from torchalg.monitoring.analysis import golub_meurant_error_bound

from neuralls.domain.solver.models.energy_error import (
    EnergyError,
    EnergyErrorBound,
    EnergyErrorUnavailable,
    ExactEnergyError,
)

if TYPE_CHECKING:
    from torchalg.models.result import SolverResult

FLOAT64_EPS = torch.finfo(torch.float64).eps
"""Machine epsilon for float64 (~2.22e-16)."""

MIN_SAFE_RELATIVE_RESIDUAL = 10 * FLOAT64_EPS
"""Floor for the reference solve's target relative residual, with headroom above the
float64 noise floor so the solve converges to a real value rather than noise."""

GOLUB_MEURANT_DELAY = 10


@dataclass(frozen=True, slots=True)
class ReferenceTarget:
    """Relative residual the reference solution must reach, and what was asked for.

    Attributes:
        requested_rel: ``rtol * margin``.
        applied_rel: The requested value, raised to ``MIN_SAFE_RELATIVE_RESIDUAL`` when it
            would ask below the float64 noise floor.
    """

    requested_rel: float
    applied_rel: float

    @property
    def is_clamped(self) -> bool:
        """Whether the request was raised to the float64 floor."""
        return self.applied_rel > self.requested_rel


def reference_target(rtol: float, margin: float) -> ReferenceTarget:
    """Relative residual a reference solution must reach for a comparison at ``rtol``.

    ``margin`` is validated to ``(0, 1)`` where the comparison config is parsed
    (``platform/config/models/comparison.py``).
    """
    requested_rel = rtol * margin
    return ReferenceTarget(requested_rel, max(requested_rel, MIN_SAFE_RELATIVE_RESIDUAL))


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

    def solve(self, b: torch.Tensor, *, target_rel: float) -> torch.Tensor:
        """Reference solution of ``A x = b`` with a checked true residual.

        Args:
            b (torch.Tensor): Right-hand side, shape ``(n,)``.
            target_rel (float): Relative residual ``||b - A x|| / ||b||`` the solution must
                reach, from ``reference_target(...).applied_rel``.

        Returns:
            torch.Tensor: Reference solution ``x*`` in float64, on the matrix's device.

        Raises:
            RuntimeError: If the true residual is not finite or exceeds the target.
        """
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


def extract_energy_error(result: SolverResult) -> EnergyError:
    """Relative energy-norm error, exact if available, else a Golub-Meurant lower bound.

    ``||v||_A = sqrt(v^T A v)`` is the norm CG minimizes, so this is the conventional
    convergence measure for (P)CG. The exact value (``result.error_history_a_norm``)
    is only populated when ``x_exact`` was passed into the solve; when it isn't (e.g.
    the reference solve failed), ``result.energy_decrements`` feeds a ground-truth-free
    lower bound on the same quantity instead.

    Args:
        result (SolverResult): Solver diagnostics returned by ``pcg``/``flexible_cg``.

    Returns:
        EnergyError: The exact curve, its bound, or why neither exists.
    """
    if result.error_history_a_norm:
        exact = normalize_error_history(result.error_history_a_norm)
        if exact is not None:
            return ExactEnergyError(exact)
        return EnergyErrorUnavailable("the initial energy error is zero")
    if result.energy_decrements:
        bound = normalize_error_history(
            golub_meurant_error_bound(result.energy_decrements, delay=GOLUB_MEURANT_DELAY)
        )
        if bound is not None:
            return EnergyErrorBound(bound)
        return EnergyErrorUnavailable("the initial energy bound is zero")
    return EnergyErrorUnavailable("the solve recorded no energy history")
