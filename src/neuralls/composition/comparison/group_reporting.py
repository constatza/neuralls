"""Report every executed system at the fine, medium and global levels.

Orchestration only: one pass collects each executed system's results and plot
styles, `classify_level` reduces them per level, and this module decides what is
logged, how each group is titled and styled, and how setup-sharing policy is
derived from the dataset caches. The destination of an actual projection is the
injected `GroupReportSink`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loguru import logger

from neuralls.composition.comparison._generation_cost import resolve_generation_repeatability
from neuralls.composition.comparison._plots import unique_display_labels
from neuralls.composition.comparison._preconditioner_setup import DatasetPreconditionerCache
from neuralls.composition.comparison.models import ComparisonOutcome, PlannedSystem
from neuralls.composition.comparison.recipes import system_tag
from neuralls.domain.benchmarking.selection import systems_solved_by_all
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.group_outcomes import GroupAggregated, GroupSkippedAllCacheHit
from neuralls.domain.solver.models.labels import FineLabel, GlobalLabel, GroupLabel, MediumLabel
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
from neuralls.domain.solver.ports import GroupReportSink
from neuralls.domain.solver.reporting.classification import classify_level
from neuralls.domain.solver.reporting.metrics_frame import SystemTag
from neuralls.domain.solver.reporting.report_levels import (
    FineLevel,
    GlobalLevel,
    MediumLevel,
    ReportLevel,
    label_key,
    report_key,
)
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.shared.types import DatasetId, GenerationRepeatability

__all__ = ["REPORT_LEVELS", "report_all_levels"]

REPORT_LEVELS: tuple[ReportLevel, ...] = (FineLevel(), MediumLevel(), GlobalLevel())
"""Every level a batch reports, finest first."""


def _sharing_by_name(
    planned: Sequence[PlannedSystem],
    specs_by_entry_id: Mapping[str, Sequence[PreconditionerConfig]],
    caches: Mapping[DatasetId, DatasetPreconditionerCache],
) -> dict[str, SetupSharingPolicy]:
    """Each preconditioner name's setup/generation sharing policy across the case.

    A name counts as shared only if every dataset that ran it shared it, so a
    projection never assumes an amortization one of its datasets did not get.
    """
    policies: dict[str, SetupSharingPolicy] = {}
    for system in planned:
        cache = caches.get(DatasetId(system.entry.matrix_dataset))
        shared_names = cache.shared_names() if cache is not None else frozenset[str]()
        for spec in specs_by_entry_id.get(system.entry.id, ()):
            policy = SetupSharingPolicy(
                share_setup=spec.name in shared_names,
                share_generation=resolve_generation_repeatability(spec)
                is GenerationRepeatability.ONE_TIME,
            )
            previous = policies.get(spec.name)
            policies[spec.name] = (
                policy
                if previous is None
                else SetupSharingPolicy(
                    share_setup=previous.share_setup and policy.share_setup,
                    share_generation=previous.share_generation and policy.share_generation,
                )
            )
    return policies


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
        f"{scope} · N = {outcome.n_aggregated} ({outcome.n_matrices} matrices × RHS)"
        f"{_common_solved_note(outcome)}{_never_converged_note(outcome)}"
    )


def _common_solved_note(outcome: GroupAggregated) -> str:
    """How many systems every preconditioner solved, when that is fewer than all of them."""
    common = len(systems_solved_by_all(outcome.outcomes))
    return f" · solved by all: {common}" if common < outcome.n_aggregated else ""


def _never_converged_note(outcome: GroupAggregated) -> str:
    """Names the preconditioners with no converged system, which have no projected curve."""
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
    """Report one batch at the fine, medium and global levels.

    Visits every report group — including one where every system was a cache
    hit — so a skipped group is logged loudly instead of silently producing
    nothing. The sink is called only for a group with a projection, so with
    every group skipped no destination is touched.

    Args:
        outcomes: Every outcome the batch is about to return.
        planned: Every planned system, in plan order.
        specs_by_entry_id: Resolved preconditioner definitions of each entry
            that executed (absent for cache hits).
        caches: Each dataset's preconditioner cache, if one was constructed
            (empty for a fully-cache-hit batch).
        sink: Receives each aggregated group's projection.
    """
    outcomes_by_id = {outcome.comparison_id: outcome for outcome in outcomes}
    tags = [system_tag(system) for system in planned]
    results_by_system: dict[SystemTag, Mapping[str, CGComparisonResult]] = {}
    styles_by_system: dict[str, Mapping[str, PreconditionerPlotStyle]] = {}
    failed_ids: set[str] = set()
    for tag in tags:
        outcome = outcomes_by_id.get(tag.system_id)
        if outcome is None:
            continue
        if outcome.payload is not None:
            results_by_system[tag] = outcome.payload.results
            styles_by_system[tag.system_id] = outcome.payload.plot_styles
        elif not outcome.success:
            failed_ids.add(tag.system_id)

    sharing = _sharing_by_name(planned, specs_by_entry_id, caches)
    for level in REPORT_LEVELS:
        for classified in classify_level(tags, results_by_system, sharing, level):
            members = [tag for tag in tags if report_key(level, tag) == label_key(classified.label)]
            n_failed = sum(tag.system_id in failed_ids for tag in members)
            match classified:
                case GroupAggregated() as aggregated:
                    if aggregated.n_aggregated < aggregated.n_in_group:
                        logger.warning(
                            "Group '{}': aggregated {} of {} systems — {}.",
                            aggregated.group_id,
                            aggregated.n_aggregated,
                            aggregated.n_in_group,
                            _unmeasured_note(
                                aggregated.n_in_group - aggregated.n_aggregated, n_failed
                            ),
                        )
                    measured = [tag for tag in members if tag in results_by_system]
                    sink.report(
                        aggregated,
                        presentation=GroupPresentation(
                            subject=_subject(aggregated.label),
                            subtitle=_subtitle(aggregated),
                            plot_styles=_plot_styles(measured, styles_by_system),
                        ),
                    )
                case GroupSkippedAllCacheHit(group_id=group_id, n_in_group=n_total):
                    logger.warning(
                        "Group '{}': nothing measured this run among {} systems — {}.",
                        group_id,
                        n_total,
                        _unmeasured_note(n_total, n_failed),
                    )
