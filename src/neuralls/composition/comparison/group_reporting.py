"""Report every executed system at the fine, medium and global levels.

Event-backed group reporting resumes when method-major execution supplies
canonical setup and solve events. Denormalized diagnostic results are not
converted back into timing events because that would duplicate matrix setup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Never, assert_never, cast

from loguru import logger

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.benchmarking.selection import NoReferenceConfigured, ReferenceSelection
from neuralls.domain.presentation.methods import DuplicateMethodPresentationError
from neuralls.domain.presentation.planner import prepared_of, realizations_of, reduce_realized_facts
from neuralls.domain.solver.models.group_outcomes import (
    GroupAggregated,
    GroupSkippedAllCacheHit,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.labels import GlobalLabel, MediumLabel
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.presentation import (
    GroupMethodPresentation,
    GroupPresentation,
    SolverTimingContext,
)
from neuralls.domain.solver.ports import GroupReportSink
from neuralls.domain.solver.reporting.classification import classify_level
from neuralls.domain.solver.reporting.report_levels import (
    FineLevel,
    GlobalLevel,
    MediumLevel,
    ReportLevel,
)
from neuralls.shared.types import PreconditionerName, SystemId

__all__ = ["REPORT_LEVELS", "report_all_levels"]

REPORT_LEVELS: tuple[ReportLevel, ...] = (FineLevel(), MediumLevel(), GlobalLevel())
"""Every level a batch reports, finest first."""


def _subject(report: RecipeProfileReport | WorkloadMixtureReport) -> str:
    """Concise scope shown beneath a figure title."""
    match report:
        case RecipeProfileReport(label=label):
            return f"Dataset {label.dataset} - {label.recipe}"
        case WorkloadMixtureReport(label=MediumLabel(dataset=dataset)):
            return f"Dataset {dataset}"
        case WorkloadMixtureReport(label=GlobalLabel()):
            return "All datasets"
        case _ as unreachable:
            assert_never(cast(Never, unreachable))


def _subtitle(outcome: RecipeProfileReport | WorkloadMixtureReport) -> str:
    """Return only the concise scope; scientific context belongs in metadata."""
    match outcome:
        case RecipeProfileReport(label=label):
            scope = f"Dataset {label.dataset} - {label.recipe}"
        case WorkloadMixtureReport(label=MediumLabel(dataset=dataset)):
            scope = f"Dataset {dataset}"
        case WorkloadMixtureReport(label=GlobalLabel()):
            scope = "All datasets"
        case _ as unreachable:
            assert_never(cast(Never, unreachable))
    return scope


def _require_distinct_labels(plot_styles: Mapping[str, PreconditionerPlotStyle]) -> None:
    """Refuse two methods that would be drawn under one legend entry.

    A shared label means the methods' scientific presentations do not distinguish them;
    appending an internal key would hide that, so it is an error.
    """
    names_by_label: dict[str, list[PreconditionerName]] = {}
    for name, style in plot_styles.items():
        names_by_label.setdefault(style.label, []).append(PreconditionerName(name))
    for names in names_by_label.values():
        if len(names) > 1:
            raise DuplicateMethodPresentationError(tuple(names))


def _group_methods(
    workloads: Sequence[CompletedMethodWorkload], system_ids: tuple[SystemId, ...]
) -> dict[PreconditionerName, GroupMethodPresentation]:
    """Describe each method with only the realized facts invariant over this group's systems."""
    members = frozenset(system_ids)
    return {
        workload.method: GroupMethodPresentation(
            prepared=prepared_of(workload.presentation),
            invariant_facts=reduce_realized_facts(
                tuple(
                    system
                    for system in realizations_of(workload.presentation)
                    if system.system_id in members
                )
            ),
        )
        for workload in workloads
    }


def report_all_levels(
    workloads: Sequence[CompletedMethodWorkload],
    sink: GroupReportSink,
    *,
    reference_selection: ReferenceSelection | dict[SystemId, ReferenceSelection] | None = None,
    contexts_by_system: Mapping[SystemId, SolverTimingContext] | None = None,
) -> None:
    """Reduce completed method workloads and report every canonical level."""
    selected = reference_selection if reference_selection is not None else NoReferenceConfigured()
    planned_by_id = {tag.system_id: tag for workload in workloads for tag in workload.planned}
    planned = tuple(planned_by_id.values())
    planned_by_method = {workload.method: workload.planned for workload in workloads}
    setup_events = tuple(event for workload in workloads for event in workload.setup_events)
    attempts = tuple(attempt for workload in workloads for attempt in workload.solve_attempts)
    plot_styles = {
        name: style for workload in workloads for name, style in workload.plot_styles.items()
    }
    _require_distinct_labels(plot_styles)
    for level in REPORT_LEVELS:
        for classified in classify_level(
            planned, setup_events, attempts, level, selected, planned_by_method=planned_by_method
        ):
            match classified:
                case GroupAggregated() as aggregated:
                    solver_contexts = tuple(
                        dict.fromkeys(
                            contexts_by_system[system_id]
                            for system_id in aggregated.system_ids
                            if contexts_by_system is not None and system_id in contexts_by_system
                        )
                    )
                    sink.report(
                        aggregated,
                        presentation=GroupPresentation(
                            subject=_subject(aggregated),
                            subtitle=_subtitle(aggregated),
                            plot_styles=plot_styles,
                            methods=_group_methods(workloads, aggregated.system_ids),
                            solver_contexts=solver_contexts,
                        ),
                    )
                case GroupSkippedAllCacheHit(group_id=group_id, n_planned_rhs=n_total):
                    logger.warning(
                        "Group '{}': no canonical setup events among {} planned systems.",
                        group_id,
                        n_total,
                    )
