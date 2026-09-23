"""Comparison diagnostic plot generation."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from neuralls.composition.comparison._presentation import build_comparison_plot_title
from neuralls.composition.comparison.models import ComparisonPaths
from neuralls.domain.solver.models.result import CGComparisonResult, PlotPaths
from neuralls.platform.config.models.preconditioner_family import PreconditionerFamilyKey
from neuralls.platform.reporting.plots import (
    plot_convergence_comparison,
    plot_error_convergence_comparison,
    plot_metric_comparison,
)


def _generate_comparison_plots(
    results: dict[str, CGComparisonResult],
    paths: ComparisonPaths,
    labels: Mapping[str, str],
    comparison_context: str,
    families: Mapping[str, PreconditionerFamilyKey] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    system_size: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iterations: int | None = None,
) -> PlotPaths:
    """Generate diagnostic plots for a comparison run.

    Args:
        results: CG comparison results keyed by preconditioner name.
        paths: Resolved comparison paths (figures directory used for output).
        labels: Descriptive plot label per preconditioner name (e.g. AMG grid
            levels/cycle/coarsening, POD-2G fitted rank), typically built via
            ``build_preconditioner_labels`` while the preconditioner is still
            constructed.
        comparison_context: Canonical matrix/RHS context shown on all plots.
        families: Plot-style family per preconditioner name (see
            ``preconditioner_family.preconditioner_family``), used to give
            same-family convergence lines a shared linestyle.
        color_keys: Optional color-axis key per preconditioner name (e.g. a
            POD-2G fit dataset); entries missing here fall back to family.
        marker_keys: Optional marker-axis key per preconditioner name (e.g. a
            POD-2G weighting scheme); entries missing here fall back to family.
        system_size: Optional linear system size ``N`` appended to plot titles.
        rtol: Relative tolerance displayed as a reference line.
        atol: Absolute tolerance displayed as a reference line.
        max_iterations: Maximum iterations displayed as a reference line.

    Returns:
        Typed PlotPaths with paths to all generated figures.
    """
    suffix = paths.matrix.stem or "comparison"
    families = families or {}
    color_keys = color_keys or {}
    marker_keys = marker_keys or {}
    title = build_comparison_plot_title(comparison_context, system_size)

    by_label = {labels.get(name, name): result for name, result in results.items()}
    family_by_label = {labels.get(name, name): value for name, value in families.items()}
    color_by_label = {labels.get(name, name): value for name, value in color_keys.items()}
    marker_by_label = {labels.get(name, name): value for name, value in marker_keys.items()}

    convergence_path = paths.figures / f"preconditioner_convergence_{suffix}.png"
    plot_convergence_comparison(
        by_label,
        metadata=None,
        save_path=convergence_path,
        title=title,
        rtol=rtol,
        atol=atol,
        max_iterations=max_iterations,
        families=family_by_label,
        color_keys=color_by_label,
        marker_keys=marker_by_label,
    )

    error_path = paths.figures / f"preconditioner_error_convergence_{suffix}.png"
    plot_error_convergence_comparison(
        by_label,
        metadata=None,
        save_path=error_path,
        title=title,
        families=family_by_label,
        color_keys=color_by_label,
        marker_keys=marker_by_label,
    )

    iter_path = paths.figures / f"preconditioner_iterations_{suffix}.png"
    plot_metric_comparison(
        [labels.get(name, name) for name in results],
        [r.iterations for r in results.values()],
        metric_name="CG Iterations",
        title=title,
        horizontal=True,
        save_path=iter_path,
    )

    return PlotPaths(
        convergence=convergence_path,
        iterations_barplot=iter_path,
        error_convergence=error_path,
    )
