"""Report every executed system at the fine, medium and global levels.

Event-backed group reporting resumes when method-major execution supplies
canonical setup and solve events. Denormalized diagnostic results are not
converted back into timing events because that would duplicate matrix setup.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from neuralls.composition.comparison.models import CompletedMethodWorkload
from neuralls.domain.benchmarking.selection import systems_solved_by_all
from neuralls.domain.solver.models.group_outcomes import GroupAggregated, GroupSkippedAllCacheHit
from neuralls.domain.solver.models.labels import FineLabel, GlobalLabel, GroupLabel, MediumLabel
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.domain.solver.ports import GroupReportSink
from neuralls.domain.solver.reporting.classification import classify_level
from neuralls.domain.solver.reporting.report_levels import (
    FineLevel,
    GlobalLevel,
    MediumLevel,
    ReportLevel,
)

__all__ = ["REPORT_LEVELS", "report_all_levels"]

REPORT_LEVELS: tuple[ReportLevel, ...] = (FineLevel(), MediumLevel(), GlobalLevel())
"""Every level a batch reports, finest first."""


def _mixture_text(mixture: Sequence[str]) -> str:
    return f"mixture of {', '.join(mixture)}"


def _subject(label: GroupLabel) -> str:
    """What a group is, for plot titles: medium and global say they mix recipes."""
    match label:
        case FineLabel(dataset=dataset, recipe=recipe):
            return f"{dataset}, {recipe}"
        case MediumLabel(dataset=dataset, mixture_of=mixture):
            return f"{dataset}, {_mixture_text(mixture)}"
        case GlobalLabel(mixture_of=mixture):
            return f"all datasets, {_mixture_text(mixture)}"


def _subtitle(outcome: GroupAggregated) -> str:
    """Level, keys, recipe (or mixture) and system count of a group, as one line."""
    match outcome.label:
        case FineLabel(recipe=recipe):
            scope = f"Fine level · keys: dataset, recipe · recipe: {recipe}"
        case MediumLabel(mixture_of=mixture):
            scope = f"Medium level · keys: dataset · {_mixture_text(mixture)}"
        case GlobalLabel(mixture_of=mixture):
            scope = f"Global level · keys: none · {_mixture_text(mixture)}"
    return (
        f"{scope} · planned RHS = {outcome.n_planned_rhs} · matrices = {outcome.n_matrices}"
        f"{_common_solved_note(outcome)}{_never_converged_note(outcome)}"
    )


def _common_solved_note(outcome: GroupAggregated) -> str:
    """How many systems every preconditioner solved, when that is fewer than all of them."""
    common = len(systems_solved_by_all(outcome.outcomes))
    return f" · solved by all: {common}" if common < outcome.n_planned_rhs else ""


def _never_converged_note(outcome: GroupAggregated) -> str:
    """Name the preconditioners with no converged system."""
    names = [name for name, c in outcome.convergence.items() if c.n_converged == 0]
    return f" · never converged: {', '.join(names)}" if names else ""


def report_all_levels(
    workloads: Sequence[CompletedMethodWorkload],
    sink: GroupReportSink,
) -> None:
    """Reduce completed method workloads and report every canonical level."""
    planned_by_id = {tag.system_id: tag for workload in workloads for tag in workload.planned}
    planned = tuple(planned_by_id.values())
    setup_events = tuple(event for workload in workloads for event in workload.setup_events)
    attempts = tuple(attempt for workload in workloads for attempt in workload.solve_attempts)
    for level in REPORT_LEVELS:
        for classified in classify_level(planned, setup_events, attempts, level):
            match classified:
                case GroupAggregated() as aggregated:
                    sink.report(
                        aggregated,
                        presentation=GroupPresentation(
                            subject=_subject(aggregated.label),
                            subtitle=_subtitle(aggregated),
                            plot_styles={},
                        ),
                    )
                case GroupSkippedAllCacheHit(group_id=group_id, n_planned_rhs=n_total):
                    logger.warning(
                        "Group '{}': no canonical setup events among {} planned systems.",
                        group_id,
                        n_total,
                    )
