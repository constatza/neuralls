"""Per-system convergence figures: every method's history on one system, one figure per metric."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.presentation.methods import MissingMethodPresentationError
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import SolveConverged, SystemTag
from neuralls.domain.solver.models.figure_scope import RecipeScope
from neuralls.domain.solver.models.plot_style import plot_style_maps
from neuralls.domain.solver.reporting.event_frames import (
    CONVERGENCE_SCHEMA,
    build_convergence_frame,
)
from neuralls.platform.reporting.figure_text import EditorialTitle, ScopeSubtitle, method_legend
from neuralls.platform.reporting.plots import ConvergenceCurve, plot_convergence_curves
from neuralls.platform.tracking.mlflow import sanitize_metric_key_segment
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId


class SystemMetric(StrEnum):
    """The convergence metric one system figure shows."""

    RESIDUAL = "residual"
    ENERGY = "energy"


_METRIC_COLUMN = {
    SystemMetric.RESIDUAL: "residual_rel",
    SystemMetric.ENERGY: "energy_error_rel",
}
_METRIC_TITLE = {
    SystemMetric.RESIDUAL: "Residual History",
    SystemMetric.ENERGY: "Energy History",
}
_METRIC_YLABEL = {
    SystemMetric.RESIDUAL: "Relative residual $\\|r_k\\| / \\|b\\|$",
    SystemMetric.ENERGY: "Relative energy error $\\|e_k\\|_A / \\|e_0\\|_A$",
}


@dataclass(frozen=True, slots=True)
class SystemFigure:
    """One drawn figure: the directory slug naming its system and the staged file."""

    system_slug: str
    metric: SystemMetric
    path: Path


def concat_convergence_frames(workloads: Sequence[CompletedMethodWorkload]) -> pl.DataFrame:
    """The convergence rows of every method workload as one table."""
    frames = [build_convergence_frame(w.method, w.diagnostics) for w in workloads]
    return pl.concat(frames) if frames else pl.DataFrame(schema=CONVERGENCE_SCHEMA)


def write_system_figures(
    workloads: Sequence[CompletedMethodWorkload],
    frame: pl.DataFrame,
    *,
    datasets: Mapping[DatasetId, DatasetPresentation],
    notation: NotationPolicy,
    root: Path,
) -> tuple[SystemFigure, ...]:
    """Draw a residual and an energy figure per system under ``root/<system slug>/``.

    A figure needs at least one method with a history for the system; the energy figure
    is also skipped when every energy value of the system is null.
    """
    methods = {w.method: w.presentation.prepared.compact for w in workloads}
    legend = method_legend({str(name): compact for name, compact in methods.items()}, notation)
    styles = {name: style for w in workloads for name, style in w.plot_styles.items()}
    maps = plot_style_maps(styles)
    converged = frozenset(
        (str(a.method), str(a.system_id))
        for w in workloads
        for a in w.solve_attempts
        if isinstance(a.outcome, SolveConverged)
    )
    tags = {tag.system_id: tag for w in workloads for tag in w.planned}
    figures: list[SystemFigure] = []
    used_slugs: set[str] = set()
    for tag in tags.values():
        dataset = datasets.get(tag.dataset)
        if dataset is None:
            raise MissingMethodPresentationError(str(tag.dataset))
        slug = _unique_slug(_system_text(tag, dataset, notation), used_slugs)
        system_frame = frame.filter(pl.col("system_id") == str(tag.system_id))
        for metric in SystemMetric:
            curves = _curves(system_frame, metric, str(tag.system_id), converged)
            if not curves:
                continue
            path = root / slug / f"{metric.value}_history.png"
            plot_convergence_curves(
                curves,
                legend=legend.marking_unconverged(
                    name for name, curve in curves.items() if not curve.converged
                ),
                save_path=path,
                title=EditorialTitle(
                    f"{_METRIC_TITLE[metric]} (matrix {int(tag.matrix_id)}, RHS {tag.rhs_ordinal})"
                ),
                subtitle=ScopeSubtitle(RecipeScope(dataset, tag.recipe), notation),
                families=maps.families,
                color_keys=maps.color_keys,
                marker_keys=maps.marker_keys,
                ylabel=_METRIC_YLABEL[metric],
            )
            figures.append(SystemFigure(slug, metric, path))
    return tuple(figures)


def _curves(
    system_frame: pl.DataFrame,
    metric: SystemMetric,
    system_id: str,
    converged: frozenset[tuple[str, str]],
) -> dict[str, ConvergenceCurve]:
    """The per-method curves of one metric on one system; methods without values are absent."""
    column = _METRIC_COLUMN[metric]
    curves: dict[str, ConvergenceCurve] = {}
    for (method,), rows in system_frame.sort("iteration").group_by("method", maintain_order=True):
        values = rows.get_column(column).drop_nulls().to_list()
        if values:
            curves[str(method)] = ConvergenceCurve(
                tuple(values), converged=(str(method), system_id) in converged
            )
    return curves


def _system_text(tag: SystemTag, dataset: DatasetPresentation, notation: NotationPolicy) -> str:
    where = ScopeSubtitle(RecipeScope(dataset, tag.recipe), notation).text
    return f"{where}, matrix {int(tag.matrix_id)}, RHS {tag.rhs_ordinal}"


def _unique_slug(text: str, used: set[str]) -> str:
    base = sanitize_metric_key_segment(text)
    slug, suffix = base, 1
    while slug in used:
        suffix += 1
        slug = f"{base}_{suffix}"
    used.add(slug)
    return slug


__all__ = [
    "SystemFigure",
    "SystemMetric",
    "concat_convergence_frames",
    "write_system_figures",
]
