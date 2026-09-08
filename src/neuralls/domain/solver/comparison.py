"""CG solver comparison runner.

This module provides pure domain logic for running CG comparisons with multiple
preconditioners using Flexible CG (FCG) uniformly, result formatting, and analysis.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torchalg import flexible_cg
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import Identity
from torchalg.utils.device import resolve_device

from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonRecommendations,
    RankedRecommendation,
)
from neuralls.shared.constants import DEFAULT_ATOL, DEFAULT_M_MAX, DEFAULT_RTOL


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
) -> dict[str, CGComparisonResult]:
    """Run CG with multiple preconditioners for comparison.

    All preconditioners use flexible_cg (FCG with Gram-Schmidt orthogonalization)
    to ensure identical mathematical treatment regardless of preconditioner type.
    This is the only fair basis for comparison: every preconditioner sees the same
    algorithm, the same initial guess, and the same system.

    Args:
        A: System matrix tensor.
        b: Right-hand side vector tensor.
        preconditioners: Dict mapping names to Preconditioner instances.
        x0: Initial guess (defaults to zero).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        maxiter: Maximum iterations.
        m_max: FCG orthogonalization restart parameter.

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

    x_exact = torch.linalg.solve(A, b)

    results: dict[str, CGComparisonResult] = {}

    for precond_name, precond in preconditioners.items():
        try:
            x_sol, info = flexible_cg(
                A,
                b,
                x0,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                preconditioner=precond,
                m_max=m_max,
            )
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
            exact_error = _relative_exact_error(x_sol, x_exact)

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
            )

        results[precond_name] = result

    return results


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    """Convert solver tensors at the reporting DTO boundary."""
    return value.detach().cpu().numpy()


def _relative_exact_error(x_sol: torch.Tensor, x_exact: torch.Tensor) -> float:
    """Relative error ``||x_sol - x_exact|| / ||x_exact||``, or absolute if ``x_exact`` is ~0.

    Aligns ``x_exact`` to ``x_sol``'s device before combining them — device
    only, never dtype, so a real precision mismatch stays visible instead of
    being silently downcast.
    """
    x_exact_matched = x_exact.to(device=x_sol.device)
    exact_norm = float(torch.linalg.vector_norm(x_exact_matched))
    diff_norm = float(torch.linalg.vector_norm(x_sol - x_exact_matched))
    if exact_norm == 0:
        return diff_norm
    return diff_norm / exact_norm


def _requires_flexible_cg(preconditioner: Preconditioner) -> bool:
    """Check if preconditioner requires flexible CG.

    Delegates to the preconditioner's own knowledge of solver compatibility,
    following OCP: adding a new preconditioner type never requires updating
    this function.

    Args:
        preconditioner: Preconditioner instance to check.

    Returns:
        True if flexible CG is required.
    """
    return preconditioner.requires_flexible_cg


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
    lines = ["Flexible CG results:"]
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
