"""Per-system convergence figures: every method's history on one system, one figure per metric."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl

from neuralls.composition.comparison.postprocessing.report_inputs import ReportedMethod
from neuralls.domain.benchmarking.selection import ReferenceSelection
from neuralls.domain.presentation.methods import MissingMethodPresentationError
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import SolveConverged, SystemTag
from neuralls.domain.solver.models.figure_scope import RecipeScope, SystemScope
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.models.plot_style import plot_style_maps
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.reporting.benchmark_events import MethodBenchmarkArtifacts
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    FigureMetadata,
    FigureMethod,
    FigurePlan,
    StagedFigure,
    method_source,
)
from neuralls.platform.reporting.figure_text import EditorialTitle, ScopeSubtitle, method_legend
from neuralls.platform.reporting.plots import ConvergenceCurve, plot_convergence_curves
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, SystemId


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
_METRIC_KIND = {
    SystemMetric.RESIDUAL: FigureKind.RESIDUAL_HISTORY,
    SystemMetric.ENERGY: FigureKind.ENERGY_HISTORY,
}
_METRIC_DEFINITION = {
    SystemMetric.RESIDUAL: "relative residual by iteration",
    SystemMetric.ENERGY: "relative energy error by iteration",
}
_METRIC_YLABEL = {
    SystemMetric.RESIDUAL: "Relative residual $\\|r_k\\| / \\|b\\|$",
    SystemMetric.ENERGY: "Relative energy error $\\|e_k\\|_A / \\|e_0\\|_A$",
}


@dataclass(frozen=True, slots=True)
class SystemFigure:
    """One drawn figure: the system it belongs to, and its staged file with the plan describing it."""

    tag: SystemTag
    metric: SystemMetric
    staged: StagedFigure


def write_system_figures(
    workloads: Sequence[ReportedMethod],
    frame: pl.DataFrame,
    *,
    datasets: Mapping[DatasetId, DatasetPresentation],
    references: Mapping[SystemId, ReferenceSelection],
    solver_contexts: Mapping[SystemId, SolverTimingContext],
    notation: NotationPolicy,
    root: Path,
) -> tuple[SystemFigure, ...]:
    """Draw a residual and an energy figure per system under ``root/system_<n>/``.

    A figure needs at least one method with a history for the system; the energy figure
    is also skipped when every energy value of the system is null. Each figure is planned
    with the methods that drew a curve, the system's solver settings and its reference.
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
    by_method = {str(w.method): w for w in workloads}
    figures: list[SystemFigure] = []
    for index, tag in enumerate(tags.values()):
        dataset = datasets.get(tag.dataset)
        if dataset is None:
            raise MissingMethodPresentationError(str(tag.dataset))
        system_frame = frame.filter(pl.col("system_id") == str(tag.system_id))
        for metric in SystemMetric:
            curves = _curves(system_frame, metric, str(tag.system_id), converged)
            if not curves:
                continue
            filename = f"{metric.value}_history.png"
            path = root / f"system_{index}" / filename
            title = f"{_METRIC_TITLE[metric]} (matrix {int(tag.matrix_id)}, RHS {tag.rhs_ordinal})"
            drawn = [by_method[name] for name in curves]
            plot_convergence_curves(
                curves,
                legend=legend.marking_unconverged(
                    name for name, curve in curves.items() if not curve.converged
                ),
                save_path=path,
                title=EditorialTitle(title),
                subtitle=ScopeSubtitle(RecipeScope(dataset, tag.recipe), notation),
                families=maps.families,
                color_keys=maps.color_keys,
                marker_keys=maps.marker_keys,
                ylabel=_METRIC_YLABEL[metric],
            )
            metadata = _metadata(
                metric,
                tag,
                dataset,
                drawn,
                reference=references[tag.system_id],
                context=solver_contexts[tag.system_id],
            )
            plan = FigurePlan(filename, title, metadata)
            figures.append(SystemFigure(tag, metric, StagedFigure(plan, path)))
    return tuple(figures)


def _metadata(
    metric: SystemMetric,
    tag: SystemTag,
    dataset: DatasetPresentation,
    drawn: Sequence[ReportedMethod],
    *,
    reference: ReferenceSelection,
    context: SolverTimingContext,
) -> FigureMetadata:
    """The scientific context of one system figure: one system, the methods it draws."""
    artifacts = MethodBenchmarkArtifacts()
    return FigureMetadata(
        kind=_METRIC_KIND[metric],
        metric_definition=_METRIC_DEFINITION[metric],
        scenario=None,
        report_level=ReportLevelName.SYSTEM,
        scope=SystemScope(dataset, tag.recipe, tag.matrix_id, tag.rhs_ordinal),
        dataset_ids=(str(tag.dataset),),
        recipes=(tag.recipe,),
        n_matrices=1,
        n_planned_rhs=1,
        reference_selection=reference,
        failure_treatment=FailureTreatment.NONE,
        timing_policy="diagnostic histories; excluded from online timing",
        methods=tuple(
            FigureMethod(method.method, method.presentation.prepared.detailed, method.config_digest)
            for method in drawn
        ),
        sources=tuple(
            method_source(method.run_id, artifacts.convergence, artifacts.solve_attempts)
            for method in drawn
        ),
        recipe_counts=(1,),
        aggregation_rule="one diagnostic history per method on this system",
        recipe_weights=(1.0,),
        solver_contexts=(context,),
    )


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


__all__ = [
    "SystemFigure",
    "SystemMetric",
    "write_system_figures",
]
