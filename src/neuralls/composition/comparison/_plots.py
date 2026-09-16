"""Comparison diagnostic plot generation."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from neuralls.composition.comparison.models import ComparisonPaths
from neuralls.domain.analysis.spectra import plot_condition_numbers
from neuralls.domain.solver.models.result import CGComparisonResult, PlotPaths
from neuralls.platform.config.models.preconditioner_family import PreconditionerFamilyKey
from neuralls.platform.reporting.plots import plot_convergence_comparison, plot_metric_comparison


def _comparison_plot_title(display_name: str | None, system_size: int | None) -> str | None:
    """Build the shared title for comparison-run diagnostic plots."""
    if system_size is None:
        return display_name
    prefix = display_name or "Comparison"
    return f"{prefix} (N={system_size})"


def _generate_comparison_plots(
    results: dict[str, CGComparisonResult],
    cond_numbers: dict[str, float],
    paths: ComparisonPaths,
    labels: Mapping[str, str],
    families: Mapping[str, PreconditionerFamilyKey] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    display_name: str | None = None,
    system_size: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iterations: int | None = None,
) -> PlotPaths:
    """Generate diagnostic plots for a comparison run.

    Args:
        results: CG comparison results keyed by preconditioner name.
        cond_numbers: Effective condition numbers keyed by preconditioner name.
        paths: Resolved comparison paths (figures directory used for output).
        labels: Descriptive plot label per preconditioner name (e.g. AMG grid
            levels/cycle/coarsening, POD-2G fitted rank), typically built via
            ``build_preconditioner_labels`` while the preconditioner is still
            constructed.
        families: Plot-style family per preconditioner name (see
            ``preconditioner_family.preconditioner_family``), used to give
            same-family convergence lines a shared linestyle.
        color_keys: Optional color-axis key per preconditioner name (e.g. a
            POD-2G fit dataset); entries missing here fall back to family.
        marker_keys: Optional marker-axis key per preconditioner name (e.g. a
            POD-2G weighting scheme); entries missing here fall back to family.
        display_name: Optional title shown on all plots.
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
    title = _comparison_plot_title(display_name, system_size)

    cond_path = plot_condition_numbers(
        {labels.get(name, name): value for name, value in cond_numbers.items()},
        save_dir=paths.figures,
        suffix=suffix,
        title=title,
        rtol=rtol,
        atol=atol,
    )

    convergence_path = paths.figures / f"preconditioner_convergence_{suffix}.png"
    plot_convergence_comparison(
        {labels.get(name, name): result for name, result in results.items()},
        metadata=None,
        save_path=convergence_path,
        title=title,
        rtol=rtol,
        atol=atol,
        max_iterations=max_iterations,
        families={labels.get(name, name): family for name, family in families.items()},
        color_keys={labels.get(name, name): key for name, key in color_keys.items()},
        marker_keys={labels.get(name, name): key for name, key in marker_keys.items()},
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
        condition_numbers=cond_path,
        iterations_barplot=iter_path,
    )
