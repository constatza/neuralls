"""Text summary and ranking of a comparison run's results."""

from __future__ import annotations

from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.recommendations import (
    ComparisonRecommendations,
    RankedRecommendation,
)
from neuralls.domain.solver.reporting.solve_time import time_to_solution


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
    return "\n".join(
        ["CG comparison results:"]
        + [_format_result_line(name, result) for name, result in results.items()]
    )


def _format_result_line(name: str, result: CGComparisonResult) -> str:
    """Format one preconditioner's result as a single summary line.

    Args:
        name: Preconditioner name.
        result: Its CG comparison result.

    Returns:
        One formatted line, e.g. ``"- jacobi  status=ok  iters=5  rel_res=1.0e-08"``.
    """
    status = "ok" if result.converged else "fail"
    line = (
        f"- {name:<18} status={status:<4} iters={result.iterations:>3}"
        f"  rel_res={result.residual:.3e}"
    )
    if result.residual_abs is not None:
        line += f" (abs={result.residual_abs:.3e})"
    if result.exact_error is not None:
        line += f"  exact_err={result.exact_error:.3e}"
    if result.error:
        line += f"  note={result.error}"
    if result.breakdown:
        line += "  note=breakdown_post_convergence" if result.converged else "  note=breakdown"
    return line


def summarize_best_combinations(
    results: dict[str, CGComparisonResult],
) -> ComparisonRecommendations:
    """Rank converged CG combinations by time-to-solution.

    Every converged run stops just below the tolerance, so its final residual
    says nothing about quality; what differs is the cost of getting there.
    Entries are ordered by total time (generation + setup + solve), then by
    iterations; an entry whose time is incomparable ranks after all timed
    ones.

    Args:
        results: CG comparison results.

    Returns:
        Typed ranked recommendations with an overall best entry.
    """
    ranked = sorted(
        (
            RankedRecommendation(
                label=label,
                iterations=info.iterations,
                residual=info.residual,
                residual_abs=info.residual_abs,
                breakdown=info.breakdown,
                total_time_seconds=time_to_solution(info),
            )
            for label, info in results.items()
            if info.converged
        ),
        key=lambda entry: (
            entry.total_time_seconds is None,
            entry.total_time_seconds or 0.0,
            entry.iterations,
        ),
    )
    return ComparisonRecommendations(
        ranked=tuple(ranked),
        overall_best=ranked[0] if ranked else None,
    )
