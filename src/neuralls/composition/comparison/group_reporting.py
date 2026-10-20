"""Report every executed system at the fine, medium and global levels.

Event-backed group reporting resumes when method-major execution supplies
canonical setup and solve events. Denormalized diagnostic results are not
converted back into timing events because that would duplicate matrix setup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loguru import logger

from neuralls.composition.comparison._plots import unique_display_labels
from neuralls.composition.comparison._preconditioner_setup import DatasetPreconditionerCache
from neuralls.composition.comparison.models import ComparisonOutcome, PlannedSystem
from neuralls.domain.benchmarking.selection import systems_solved_by_all
from neuralls.domain.solver.models.benchmark_events import SystemTag
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.labels import FineLabel, GlobalLabel, GroupLabel, MediumLabel
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.ports import GroupReportSink
from neuralls.domain.solver.reporting.report_levels import (
    FineLevel,
    GlobalLevel,
    MediumLevel,
    ReportLevel,
)
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.shared.types import DatasetId

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


def _unmeasured_note(n_unmeasured: int, n_failed: int) -> str:
    """Why systems produced no result: cache hits carry none, and failures never had one."""
    return f"{n_unmeasured - n_failed} cache hit(s), {n_failed} failed"


def _plot_styles(
    members: Sequence[SystemTag],
    styles_by_system: Mapping[str, Mapping[str, PreconditionerPlotStyle]],
) -> dict[str, PreconditionerPlotStyle]:
    """Style of each preconditioner across a group, first system's winning, labels disambiguated."""
    merged: dict[str, PreconditionerPlotStyle] = {}
    for tag in members:
        for name, style in styles_by_system.get(tag.system_id, {}).items():
            merged.setdefault(name, style)
    unique = unique_display_labels(merged, {name: style.label for name, style in merged.items()})
    return {
        name: PreconditionerPlotStyle(
            label=unique[name],
            family=style.family,
            color_key=style.color_key,
            marker_key=style.marker_key,
        )
        for name, style in merged.items()
    }


def report_all_levels(
    outcomes: Sequence[ComparisonOutcome],
    planned: Sequence[PlannedSystem],
    specs_by_entry_id: Mapping[str, Sequence[PreconditionerConfig]],
    caches: Mapping[DatasetId, DatasetPreconditionerCache],
    sink: GroupReportSink,
) -> None:
    """Defer output until Task 4 supplies canonical method workloads.

    Reconstructing events here would copy one matrix setup into every sibling
    RHS result, which is the cost-model defect this migration removes.
    """
    del outcomes, planned, specs_by_entry_id, caches, sink
    logger.warning("Group reporting deferred until canonical method workloads are available.")
