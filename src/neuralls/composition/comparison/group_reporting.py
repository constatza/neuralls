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
from neuralls.domain.presentation.planner import reduce_realized_facts
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.figure_scope import (
    AllDatasetsScope,
    DatasetScope,
    GroupScope,
    RecipeScope,
)
from neuralls.domain.solver.models.group_outcomes import (
    GroupAggregated,
    GroupSkippedAllCacheHit,
)
from neuralls.domain.solver.models.group_presentation import (
    GroupMethodPresentation,
    GroupPresentation,
    SolverTimingContext,
)
from neuralls.domain.solver.models.labels import FineLabel, GlobalLabel, GroupLabel, MediumLabel
from neuralls.domain.solver.ports import GroupReportSink
from neuralls.domain.solver.reporting.classification import classify_level
from neuralls.domain.solver.reporting.report_levels import (
    FineLevel,
    GlobalLevel,
    MediumLevel,
    ReportLevel,
)
from neuralls.platform.reporting.figure_text import scope_label
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, PreconditionerName, SystemId

__all__ = ["REPORT_LEVELS", "report_all_levels"]

REPORT_LEVELS: tuple[ReportLevel, ...] = (FineLevel(), MediumLevel(), GlobalLevel())
"""Every level a batch reports, finest first."""


class MissingDatasetPresentationError(LookupError):
    """A report group names a dataset whose scientific presentation was not supplied.

    The dataset id is a diagnostic machine identifier only; it never becomes a label.
    """

    def __init__(self, dataset: DatasetId) -> None:
        self.dataset = dataset
        super().__init__(f"no scientific presentation for dataset {dataset!r}")


def _scope(label: GroupLabel, datasets: Mapping[DatasetId, DatasetPresentation]) -> GroupScope:
    """What the group pools, as typed values rendered by the sink."""

    def presentation_of(dataset: DatasetId) -> DatasetPresentation:
        found = datasets.get(dataset)
        if found is None:
            raise MissingDatasetPresentationError(dataset)
        return found

    match label:
        case FineLabel(dataset=dataset, recipe=recipe):
            return RecipeScope(presentation_of(dataset), recipe)
        case MediumLabel(dataset=dataset):
            return DatasetScope(presentation_of(dataset))
        case GlobalLabel():
            return AllDatasetsScope()
        case _ as unreachable:
            assert_never(cast(Never, unreachable))


def _group_methods(
    workloads: Sequence[CompletedMethodWorkload], system_ids: tuple[SystemId, ...]
) -> dict[PreconditionerName, GroupMethodPresentation]:
    """Describe each method that has systems in this group, with its invariant realized facts.

    Realized facts are invariant only when every system of the method in this group was
    realized and they agree; a method with no system in the group is not listed.
    """
    members = frozenset(system_ids)
    methods: dict[PreconditionerName, GroupMethodPresentation] = {}
    for workload in workloads:
        population = frozenset(
            tag.system_id for tag in workload.planned if tag.system_id in members
        )
        if not population:
            continue
        methods[workload.method] = GroupMethodPresentation(
            prepared=workload.presentation.prepared,
            invariant_facts=reduce_realized_facts(
                tuple(
                    system
                    for system in workload.presentation.systems
                    if system.system_id in population
                ),
                population,
            ),
        )
    return methods


def report_all_levels(
    workloads: Sequence[CompletedMethodWorkload],
    sink: GroupReportSink,
    *,
    datasets: Mapping[DatasetId, DatasetPresentation],
    notation: NotationPolicy,
    reference_selection: ReferenceSelection | dict[SystemId, ReferenceSelection] | None = None,
    contexts_by_system: Mapping[SystemId, SolverTimingContext] | None = None,
) -> None:
    """Reduce completed method workloads and report every canonical level.

    ``datasets`` presents each dataset a group names; a group on a dataset without an
    entry raises before anything is reported for it. ``notation`` is the case's symbol
    policy, used for the text of a skipped group.
    """
    selected = reference_selection if reference_selection is not None else NoReferenceConfigured()
    planned_by_id = {tag.system_id: tag for workload in workloads for tag in workload.planned}
    planned = tuple(planned_by_id.values())
    planned_by_method = {workload.method: workload.planned for workload in workloads}
    setup_events = tuple(event for workload in workloads for event in workload.setup_events)
    attempts = tuple(attempt for workload in workloads for attempt in workload.solve_attempts)
    plot_styles = {
        name: style for workload in workloads for name, style in workload.plot_styles.items()
    }
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
                            scope=_scope(aggregated.label, datasets),
                            plot_styles=plot_styles,
                            methods=_group_methods(workloads, aggregated.system_ids),
                            solver_contexts=solver_contexts,
                        ),
                    )
                case GroupSkippedAllCacheHit(label=label, n_planned_rhs=n_total):
                    logger.warning(
                        "{}: no canonical setup events among {} planned systems.",
                        scope_label(_scope(label, datasets), notation),
                        n_total,
                    )
