"""Comparison diagnostic plot generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from neuralls.composition.comparison._presentation import build_comparison_plot_title
from neuralls.composition.comparison.models import ComparisonPaths
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.cost_fields import COST_FIELD_REGISTRY
from neuralls.domain.solver.models.plot_paths import PlotPaths
from neuralls.platform.reporting.plots import (
    plot_convergence_comparison,
    plot_error_convergence_comparison,
    plot_error_vs_time,
    plot_metric_comparison,
    plot_residual_vs_time,
    plot_time_breakdown_barplot,
    plot_work_precision,
)


def unique_display_labels(
    result_keys: Mapping[str, object], labels: Mapping[str, str]
) -> dict[str, str]:
    """Keep friendly labels, suffixing only collisions with their stable result key."""
    resolved = {key: labels.get(key, key) for key in result_keys}
    counts = Counter(resolved.values())
    return {
        key: label if counts[label] == 1 else f"{label} [{key}]" for key, label in resolved.items()
    }


DID_NOT_CONVERGE_MARK = "[did not converge]"
"""Suffix on a failed run's label, so a run stopped by the iteration cap is never read
as a cheap solve in the cost, iteration and time plots."""


def mark_unconverged(
    results: Mapping[str, CGComparisonResult], labels: Mapping[str, str]
) -> dict[str, str]:
    """Suffix the label of every run that did not converge with `DID_NOT_CONVERGE_MARK`."""
    return {
        key: label if results[key].converged else f"{label} {DID_NOT_CONVERGE_MARK}"
        for key, label in labels.items()
    }


@dataclass(frozen=True, slots=True)
class CostPlotStyle:
    """Axis label and filename stem of one cost bar chart."""

    metric_name: str
    filename_suffix: str


COST_PLOT_STYLE: Mapping[str, CostPlotStyle] = {
    "generation_time_seconds": CostPlotStyle("Generation Time (s)", "generation_time"),
    "setup_time_seconds": CostPlotStyle("Setup Time (s)", "setup_time"),
    "solve_time_seconds": CostPlotStyle("Solve Time (s)", "solve_time"),
    "avg_iteration_time_seconds": CostPlotStyle(
        "Solve Time Per Iteration (s)", "avg_iteration_time"
    ),
    "peak_memory_bytes": CostPlotStyle("Peak Memory (bytes)", "peak_memory"),
}
"""Presentation of each `COST_FIELD_REGISTRY` entry, keyed by its `field_name`."""


# TODO(urgent): Inject a narrow reporter from the I/O shell so comparison planning and
# orchestration can run without constructing output paths or rendering plot files.
def _generate_comparison_plots(
    results: dict[str, CGComparisonResult],
    paths: ComparisonPaths,
    labels: Mapping[str, str],
    comparison_context: str,
    families: Mapping[str, Hashable] | None = None,
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
        Typed PlotPaths with paths to all generated figures — cost bar charts
        (``generation_time_barplot``/``setup_time_barplot``/``solve_time_barplot``/
        ``peak_memory_barplot``/``time_breakdown_barplot``) are omitted (left
        ``None``) if no result in this comparison has that metric measured.
    """
    suffix = paths.matrix.stem or "comparison"
    labels = mark_unconverged(results, unique_display_labels(results, labels))
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

    cost_plot_paths: dict[str, Path | None] = {
        spec.field_name: _plot_cost_barplot(
            results,
            labels,
            paths.figures
            / f"preconditioner_{COST_PLOT_STYLE[spec.field_name].filename_suffix}_{suffix}.png",
            extract=spec.extractor,
            metric_name=COST_PLOT_STYLE[spec.field_name].metric_name,
            title=title,
        )
        for spec in COST_FIELD_REGISTRY
    }

    time_breakdown_path = _plot_time_breakdown_barplot(
        results, labels, paths.figures / f"preconditioner_time_breakdown_{suffix}.png", title=title,
    )  # fmt: skip

    work_precision_path = paths.figures / f"preconditioner_work_precision_{suffix}.png"
    plot_work_precision(
        results,
        labels=labels,
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
        save_path=work_precision_path,
        title=title,
    )

    residual_vs_time_path = paths.figures / f"preconditioner_residual_vs_time_{suffix}.png"
    plot_residual_vs_time(
        results,
        labels=labels,
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
        save_path=residual_vs_time_path,
        title=title,
    )

    residual_vs_solve_time_path = (
        paths.figures / f"preconditioner_residual_vs_solve_time_{suffix}.png"
    )
    plot_residual_vs_time(
        results,
        labels=labels,
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
        save_path=residual_vs_solve_time_path,
        title=title,
        solve_only=True,
    )

    error_vs_time_path = _plot_error_vs_time_optional(
        results,
        labels,
        paths.figures / f"preconditioner_error_vs_time_{suffix}.png",
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
        title=title,
    )
    error_vs_solve_time_path = _plot_error_vs_time_optional(
        results,
        labels,
        paths.figures / f"preconditioner_error_vs_solve_time_{suffix}.png",
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
        title=title,
        solve_only=True,
    )

    return PlotPaths(
        convergence=convergence_path,
        iterations_barplot=iter_path,
        error_convergence=error_path,
        generation_time_barplot=cost_plot_paths["generation_time_seconds"],
        setup_time_barplot=cost_plot_paths["setup_time_seconds"],
        solve_time_barplot=cost_plot_paths["solve_time_seconds"],
        avg_iteration_time_barplot=cost_plot_paths["avg_iteration_time_seconds"],
        peak_memory_barplot=cost_plot_paths["peak_memory_bytes"],
        time_breakdown_barplot=time_breakdown_path,
        work_precision=work_precision_path,
        residual_vs_time=residual_vs_time_path,
        error_vs_time=error_vs_time_path,
        residual_vs_solve_time=residual_vs_solve_time_path,
        error_vs_solve_time=error_vs_solve_time_path,
    )


def _plot_cost_barplot(
    results: Mapping[str, CGComparisonResult],
    labels: Mapping[str, str],
    save_path: Path,
    *,
    extract: Callable[[CGComparisonResult], float | None],
    metric_name: str,
    title: str,
) -> Path | None:
    """Bar chart of one cost value (read via ``extract``), skipping unmeasured entries.

    ``extract`` reads a plain numeric field directly (e.g. ``solve_time_seconds``)
    or unwraps a ``StageCost``'s ``wall_time_seconds`` (e.g. generation/setup
    cost) — either way this function only ever sees the resolved float.

    Returns ``None`` (no file written) when no result in this comparison has
    the value measured — e.g. every preconditioner failed to build.
    """
    present = [name for name in results if extract(results[name]) is not None]
    if not present:
        return None
    plot_metric_comparison(
        [labels.get(name, name) for name in present],
        [cast(float, extract(results[name])) for name in present],
        metric_name=metric_name,
        title=title,
        horizontal=True,
        save_path=save_path,
    )
    return save_path


def _plot_error_vs_time_optional(
    results: Mapping[str, CGComparisonResult],
    labels: Mapping[str, str],
    save_path: Path,
    *,
    families: Mapping[str, Hashable] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    title: str | None = None,
    solve_only: bool = False,
) -> Path | None:
    """Error vs time plot, skipped if no result has energy-norm error history.

    Returns ``None`` (no file written) when no result has error_history_a_rel
    measured — this is optional per comparison run.
    """
    present = [name for name in results if results[name].error_history_a_rel is not None]
    if not present:
        return None
    plot_error_vs_time(
        {name: results[name] for name in present},
        labels=labels,
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
        save_path=save_path,
        title=title,
        solve_only=solve_only,
    )
    return save_path


def _plot_time_breakdown_barplot(
    results: Mapping[str, CGComparisonResult],
    labels: Mapping[str, str],
    save_path: Path,
    *,
    title: str,
) -> Path | None:
    """Stacked generation/setup/solve time bar chart, skipping if nothing was ever measured.

    ``setup_cost`` is always a real measured float now, so only
    ``generation_cost``/``solve_time_seconds`` (both still optional) decide
    whether anything beyond setup was ever measured for a given result.

    Returns ``None`` (no file written) when no result in this comparison has
    any time component measured — mirrors ``_plot_cost_barplot``'s skip
    convention.
    """
    present = [
        name
        for name in results
        if results[name].generation_cost is not None or results[name].solve_time_seconds is not None
    ]
    if not present:
        return None
    plot_time_breakdown_barplot(
        {name: results[name] for name in present},
        labels,
        title=title,
        save_path=save_path,
    )
    return save_path
