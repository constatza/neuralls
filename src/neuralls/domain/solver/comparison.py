"""CG solver comparison runner.

This module provides pure domain logic for running CG comparisons across multiple
preconditioners — dispatching each to the cheapest CG variant it is mathematically
compatible with (standard PCG for deterministic SPD preconditioners, Flexible CG
for iteration-varying or non-SPD ones) — plus result formatting and analysis.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import numpy as np
import torch
from loguru import logger
from torchalg import flexible_cg, pcg
from torchalg.models.result import SolverResult
from torchalg.monitoring import TraceMode
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import Identity
from torchalg.utils.device import resolve_device

from neuralls.domain.solver.error_metrics import (
    REFERENCE_PRECISION_MARGIN,
    extract_energy_error,
    reference_solution,
    relative_exact_error,
)
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonRecommendations,
    RankedRecommendation,
)
from neuralls.shared.constants import DEFAULT_ATOL, DEFAULT_M_MAX, DEFAULT_RTOL


class CGAlgorithm(StrEnum):
    """CG variant a preconditioner is mathematically compatible with."""

    PCG = "pcg"
    FCG = "fcg"


def run_cg_comparison(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    preconditioners: Mapping[str, Preconditioner],
    x0: torch.Tensor | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    maxiter: int = 100,
    m_max: int = DEFAULT_M_MAX,
    reference_precision_margin: float = REFERENCE_PRECISION_MARGIN,
) -> dict[str, CGComparisonResult]:
    """Run CG with multiple preconditioners for comparison.

    Each preconditioner is routed to the CG variant it is mathematically
    compatible with: standard PCG's two-term recurrence for preconditioners
    that are deterministic and exactly SPD for the whole solve
    (``Preconditioner.requires_flexible_cg`` is ``False``), or Flexible CG's
    explicit Gram-Schmidt orthogonalization for preconditioners that vary per
    iteration or are not exactly SPD (neural preconditioners, non-linear AMG,
    ``ScheduledPreconditioner``). This keeps the comparison fair by giving
    every preconditioner the correct — and cheapest — algorithm for its own
    mathematical properties, rather than forcing a single algorithm on all of
    them; every preconditioner still sees the same initial guess and system.

    Args:
        A: System matrix tensor.
        b: Right-hand side vector tensor.
        preconditioners: Dict mapping names to Preconditioner instances.
        x0: Initial guess (defaults to zero).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        maxiter: Maximum iterations.
        m_max: FCG orthogonalization window, used only for preconditioners
            routed to ``flexible_cg``; ignored for preconditioners routed to
            ``pcg``.
        reference_precision_margin: How many orders of magnitude tighter than
            ``rtol`` the host reference solution must be.

    Returns:
        Dict mapping preconditioner names to CGComparisonResult.

    Example:
        >>> from torchalg.preconditioners.implementations import Identity, JacobiPreconditioner
        >>> preconditioners = {
        ...     "none": Identity(),
        ...     "jacobi": JacobiPreconditioner(A),
        ... }
        >>> results = run_cg_comparison(A, b, preconditioners=preconditioners)
    """
    device = resolve_device()
    A = A.to(device)
    b = b.to(device)
    x0 = torch.zeros_like(b, dtype=A.dtype, device=A.device) if x0 is None else x0.to(device)
    x0_base = x0.detach().clone()

    if "none" not in preconditioners:
        preconditioners = dict(preconditioners)
        preconditioners["none"] = Identity()

    x_exact_host = _host_reference_solution(
        A.detach().cpu(), b.detach().cpu(), rtol=rtol, margin=reference_precision_margin
    )
    x_exact = x_exact_host.to(device=A.device, dtype=A.dtype) if x_exact_host is not None else None

    results: dict[str, CGComparisonResult] = {}

    for precond_name, precond in preconditioners.items():
        try:
            x_sol, info = _solve_one(
                A, b, x0, precond, rtol=rtol, atol=atol, maxiter=maxiter, m_max=m_max,
                x_exact=x_exact,
            )  # fmt: skip
        except (ValueError, RuntimeError) as solver_exc:
            result = CGComparisonResult(
                x=_to_numpy(x0_base),
                converged=False,
                iterations=0,
                residual=float("inf"),
                residual_abs=float("inf"),
                residual_history_rel=[],
                residual_history_abs=[],
                preconditioner=precond_name,
                initial_guess=_to_numpy(x0_base),
                exact_error=None,
                rhs_norm=float(torch.linalg.vector_norm(b)),
                breakdown=False,
                error=f"CG solver failed: {solver_exc}",
            )
        else:
            exact_error = relative_exact_error(x_sol, x_exact) if x_exact is not None else None
            error_history, error_bound = _safe_extract_energy_error(info)

            rhs_norm = info.rhs_norm
            residual: list[float] = list(info.residual_history_abs or (info.residual_abs,))
            residual_rel: list[float] = (
                [r / rhs_norm for r in residual] if rhs_norm > 0 else list(residual)
            )

            result = CGComparisonResult(
                x=_to_numpy(x_sol),
                converged=info.converged,
                iterations=info.iterations,
                residual=info.residual,
                residual_abs=info.residual_abs,
                residual_history_rel=residual_rel,
                residual_history_abs=residual,
                preconditioner=precond_name,
                initial_guess=_to_numpy(x0_base),
                exact_error=exact_error,
                rhs_norm=info.rhs_norm,
                breakdown=info.breakdown,
                error_history_a_rel=error_history,
                error_bound_a_rel=error_bound,
            )

        results[precond_name] = result
        release_device_memory()

    return results


def _host_reference_solution(
    A_host: torch.Tensor, b_host: torch.Tensor, *, rtol: float, margin: float
) -> torch.Tensor | None:
    """Reference solution computed on the host so it never competes for GPU memory.

    Returns:
        The reference solution, or ``None`` (logged) if it cannot be computed —
        the comparison then runs without exact-error metrics instead of failing.
    """
    try:
        return reference_solution(A_host, b_host, rtol=rtol, margin=margin)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Reference solution unavailable; exact-error metrics skipped: {}", exc)
        return None


def _safe_extract_energy_error(
    info: SolverResult,
) -> tuple[list[float] | None, list[float] | None]:
    """Resilience wrapper around ``extract_energy_error``.

    Returns:
        ``(error_history_a_rel, error_bound_a_rel)``, or ``(None, None)`` if
        extraction fails — a metrics failure must never discard an otherwise
        valid solve.
    """
    try:
        return extract_energy_error(info)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Energy-norm error metrics skipped: {}", exc)
        return None, None


def release_device_memory() -> None:
    """Return cached CUDA blocks between preconditioners (no-op without CUDA)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    """Convert solver tensors at the reporting DTO boundary."""
    return value.detach().cpu().numpy()


def _cg_algorithm_for(preconditioner: Preconditioner) -> CGAlgorithm:
    """Classify which CG variant a preconditioner is compatible with.

    Delegates to the preconditioner's own knowledge of solver compatibility,
    following OCP: adding a new preconditioner type never requires updating
    this function.

    Args:
        preconditioner: Preconditioner instance to classify.

    Returns:
        ``CGAlgorithm.FCG`` if flexible CG is required, else ``CGAlgorithm.PCG``.
    """
    return CGAlgorithm.FCG if preconditioner.requires_flexible_cg else CGAlgorithm.PCG


def _solve_one(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    preconditioner: Preconditioner,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
    m_max: int,
    x_exact: torch.Tensor | None,
) -> tuple[torch.Tensor, SolverResult]:
    """Solve with the CG variant this preconditioner's category requires.

    Always traces at ``TraceMode.MINIMAL`` (scalar residual history only, no
    per-iterate vectors). When ``x_exact`` is supplied, ``torchalg`` tracks the
    exact energy-norm error every iteration from it directly — no vector storage,
    no separate FULL-trace pass needed for that metric.

    Args:
        A: System matrix tensor.
        b: Right-hand side vector tensor.
        x0: Initial guess tensor.
        preconditioner: Preconditioner instance to solve with.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        maxiter: Maximum iterations.
        m_max: FCG orthogonalization window; unused on the PCG branch — PCG's
            own (unrelated, off-by-default) periodic-reorthogonalization
            ``m_max`` is intentionally left disabled here.
        x_exact: Reference solution (already on ``A``'s device/dtype), or
            ``None`` if unavailable.

    Returns:
        Tuple of the solved vector and solver diagnostics.
    """
    match _cg_algorithm_for(preconditioner):
        case CGAlgorithm.PCG:
            return pcg(
                A,
                b,
                x0,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                preconditioner=preconditioner,
                trace_mode=TraceMode.MINIMAL,
                x_exact=x_exact,
            )
        case CGAlgorithm.FCG:
            return flexible_cg(
                A,
                b,
                x0,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                preconditioner=preconditioner,
                m_max=m_max,
                trace_mode=TraceMode.MINIMAL,
                x_exact=x_exact,
            )


def format_results_summary(results: dict[str, CGComparisonResult]) -> str:
    """Format CG results into a readable summary.

    Status codes follow SciPy convention:
    - "ok": converged (info=0)
    - "fail": did not converge (info>0), either max iterations or breakdown

    Args:
        results: CG comparison results.

    Returns:
        Formatted summary string.
    """
    lines = ["CG comparison results:"]
    for name, result in results.items():
        status = "ok" if result.converged else "fail"
        iters = result.iterations
        res = result.residual
        exact_err = result.exact_error
        res_abs = result.residual_abs

        if exact_err is not None:
            line = f"- {name:<18} status={status:<4} iters={iters:>3}  rel_res={res:.3e}"
            if res_abs is not None:
                line += f" (abs={res_abs:.3e})"
            line += f"  exact_err={exact_err:.3e}"
        else:
            line = f"- {name:<18} status={status:<4} iters={iters:>3}  rel_res={res:.3e}"
            if res_abs is not None:
                line += f" (abs={res_abs:.3e})"

        if result.error:
            line += f"  note={result.error}"

        if result.breakdown and not result.converged:
            line += "  note=breakdown"
        elif result.breakdown and result.converged:
            line += "  note=breakdown_post_convergence"

        lines.append(line)

    return "\n".join(lines)


def summarize_best_combinations(
    results: dict[str, CGComparisonResult],
) -> ComparisonRecommendations:
    """Summarize best-performing CG combinations by preconditioner and overall.

    Args:
        results: CG comparison results.

    Returns:
        Typed ranked recommendations with an overall best entry.
    """
    ranked: list[RankedRecommendation] = []
    for label, info in results.items():
        if not info.converged:
            continue
        ranked.append(
            RankedRecommendation(
                label=label,
                iterations=info.iterations,
                residual=info.residual,
                residual_abs=info.residual_abs,
                breakdown=info.breakdown,
            )
        )

    ranked = sorted(ranked, key=lambda entry: entry.residual)
    overall_best = ranked[0] if ranked else None

    return ComparisonRecommendations(
        ranked=tuple(ranked),
        overall_best=overall_best,
    )
