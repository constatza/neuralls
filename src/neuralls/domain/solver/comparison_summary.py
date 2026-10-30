"""Text summary and ranking of a comparison run's results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    MissingMethodPresentationError,
)
from neuralls.domain.presentation.render import render_distinct_methods
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.shared.scientific_description.policy import (
    LOG_PROFILE,
    NotationPolicy,
    RenderTarget,
)

_LABEL_WIDTH = 18


def format_results_summary(
    results: dict[str, CGComparisonResult],
    methods: Mapping[str, ConfiguredMethodPresentation],
    notation: NotationPolicy,
) -> str:
    """Format CG results into a readable summary.

    Status codes follow SciPy convention:
    - "ok": converged (info=0)
    - "fail": did not converge (info>0), either max iterations or breakdown

    Each line is headed by the method's scientific description, rendered here from
    ``methods``; a result without a presentation raises instead of falling back to its key.

    Args:
        results: CG comparison results, keyed by method id.
        methods: The presentation of every method in ``results``.
        notation: The case's symbol policy.

    Returns:
        Formatted summary string.

    Raises:
        MissingMethodPresentationError: When a result has no presentation.
    """
    missing = next((name for name in results if name not in methods), None)
    if missing is not None:
        raise MissingMethodPresentationError(missing)
    texts = render_distinct_methods(
        {name: methods[name] for name in results},
        LOG_PROFILE,
        replace(notation, target=RenderTarget.PLAIN),
    ).texts
    return "\n".join(
        ["CG comparison results:"]
        + [_format_result_line(texts[name].text, result) for name, result in results.items()]
    )


def _format_result_line(label: str, result: CGComparisonResult) -> str:
    """Format one method's result as a single summary line.

    Args:
        label: The method's rendered description.
        result: Its CG comparison result.

    Returns:
        One formatted line, e.g. ``"- Jacobi  status=ok  iters=5  rel_res=1.0e-08"``.
    """
    status = "ok" if result.converged else "fail"
    line = (
        f"- {label:<{_LABEL_WIDTH}} status={status:<4} iters={result.iterations:>3}"
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
