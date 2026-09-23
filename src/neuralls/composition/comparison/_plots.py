"""Comparison diagnostic plot generation."""

from __future__ import annotations

from collections import Counter
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


def _unique_display_labels(
    result_keys: Mapping[str, object], labels: Mapping[str, str]
) -> dict[str, str]:
    """Keep friendly labels, suffixing only collisions with their stable result key."""
    resolved = {key: labels.get(key, key) for key in result_keys}
    counts = Counter(resolved.values())
    return {
        key: label if counts[label] == 1 else f"{label} [{key}]" for key, label in resolved.items()
    }


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
        labels: Descriptive display label per preconditioner key (e.g. AMG
            grid/coarsening detail or POD-2G fitted rank and fit dataset).
            Labels are passed to reporting separately and are never used as
            result identity, so they need not be unique.
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
    labels = _unique_display_labels(results, labels)
    families = families or {}
    color_keys = color_keys or {}
    marker_keys = marker_keys or {}
    title = build_comparison_plot_title(comparison_context, system_size)

    convergence_path = paths.figures / f"preconditioner_convergence_{suffix}.png"
    plot_convergence_comparison(
        results,
        metadata=None,
        labels=labels,
        save_path=convergence_path,
        title=title,
        rtol=rtol,
        atol=atol,
        max_iterations=max_iterations,
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
    )

    error_path = paths.figures / f"preconditioner_error_convergence_{suffix}.png"
    plot_error_convergence_comparison(
        results,
        metadata=None,
        labels=labels,
        save_path=error_path,
        title=title,
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
    )

    iter_path = paths.figures / f"preconditioner_iterations_{suffix}.png"
    plot_metric_comparison(
        [labels.get(name, name) for name in results],
        [results[name].iterations for name in results],
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
