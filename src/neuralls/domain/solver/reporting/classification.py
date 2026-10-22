"""Reduce canonical benchmark events to one closed outcome per report group."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from neuralls.domain.benchmarking.reference_comparison import compare_to_reference
from neuralls.domain.benchmarking.selection import NoReferenceConfigured, ReferenceSelection
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    SolveAttempt,
    SystemTag,
    UnknownPeakMemory,
)
from neuralls.domain.solver.models.group_outcomes import (
    ComparisonGroupOutcome,
    GroupSkippedAllCacheHit,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.labels import (
    FineLabel,
    GlobalLabel,
    MediumLabel,
    render_group_id,
)
from neuralls.domain.solver.reporting.aggregation import (
    convergence_by_level,
    outcomes_by_system,
    summarize_method_workload,
)
from neuralls.domain.solver.reporting.report_levels import ReportLevel, label_for_tags, report_key
from neuralls.shared.constants import DEFAULT_RANDOM_SEED
from neuralls.shared.types import PreconditionerName, SystemId


def classify_level(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
    level: ReportLevel,
    selection: ReferenceSelection | Mapping[SystemId, ReferenceSelection] | None = None,
    *,
    planned_by_method: Mapping[PreconditionerName, Sequence[SystemTag]] | None = None,
) -> tuple[ComparisonGroupOutcome, ...]:
    """Classify planned groups from linked setup events and solve attempts."""
    convergence = convergence_by_level(
        planned, setup_events, attempts, level, planned_by_method=planned_by_method
    )
    outcomes = outcomes_by_system(
        planned, setup_events, attempts, level, planned_by_method=planned_by_method
    )
    tags_by_key: dict[tuple[str, ...], list[SystemTag]] = {}
    for tag in planned:
        tags_by_key.setdefault(report_key(level, tag), []).append(tag)

    classified: list[ComparisonGroupOutcome] = []
    for tags in tags_by_key.values():
        selected = _selection_for_tags(tags, selection)
        label = label_for_tags(level, tags)
        n_matrices = len({(tag.dataset, tag.matrix_id) for tag in tags})
        convergence_by_method = convergence.get(label)
        if convergence_by_method is None:
            classified.append(
                GroupSkippedAllCacheHit(
                    group_id=render_group_id(label),
                    label=label,
                    n_planned_rhs=len(tags),
                    n_matrices=n_matrices,
                )
            )
            continue
        recipe_membership = tuple(dict.fromkeys(tag.recipe for tag in tags))
        dataset_membership = tuple(dict.fromkeys(tag.dataset for tag in tags))
        recipe_counts = tuple(
            sum(tag.recipe == recipe for tag in tags) for recipe in recipe_membership
        )
        system_ids = tuple(tag.system_id for tag in tags)
        methods = tuple(convergence_by_method)
        group_matrices = {(tag.dataset, tag.matrix_id) for tag in tags}
        group_systems = {tag.system_id for tag in tags}
        summaries = tuple(
            summarize_method_workload(
                method,
                tags,
                tuple(
                    event
                    for event in setup_events
                    if event.method == method and (event.dataset, event.matrix_id) in group_matrices
                ),
                tuple(
                    attempt
                    for attempt in attempts
                    if attempt.method == method and attempt.system_id in group_systems
                ),
                UnknownPeakMemory(),
                allocation_planned=planned,
            )
            for method in methods
        )
        match label:
            case FineLabel():
                group_outcomes = {
                    method: outcomes.get(label, {}).get(method, {})
                    for method in convergence_by_method
                }
                comparisons = compare_to_reference(
                    group_outcomes, selection=selected, seed=DEFAULT_RANDOM_SEED
                )
                classified.append(
                    RecipeProfileReport(
                        group_id=render_group_id(label),
                        label=label,
                        n_matrices=n_matrices,
                        n_planned_rhs=len(tags),
                        convergence=convergence_by_method,
                        outcomes=group_outcomes,
                        comparisons=comparisons,
                        summaries=summaries,
                        reference_selection=selected,
                        recipe_membership=recipe_membership,
                        dataset_membership=dataset_membership,
                        recipe_counts=recipe_counts,
                        system_ids=system_ids,
                    )
                )
            case MediumLabel() | GlobalLabel():
                classified.append(
                    WorkloadMixtureReport(
                        group_id=render_group_id(label),
                        label=label,
                        n_matrices=n_matrices,
                        n_planned_rhs=len(tags),
                        convergence=convergence_by_method,
                        summaries=summaries,
                        reference_selection=selected,
                        recipe_membership=recipe_membership,
                        dataset_membership=dataset_membership,
                        recipe_counts=recipe_counts,
                        system_ids=system_ids,
                    )
                )
    return tuple(classified)


def _selection_for_tags(
    tags: Sequence[SystemTag],
    selection: ReferenceSelection | Mapping[SystemId, ReferenceSelection] | None,
) -> ReferenceSelection:
    """Resolve one report group's explicit selection without observing outcomes."""
    if selection is None:
        return NoReferenceConfigured()
    if not isinstance(selection, Mapping):
        return selection
    selections = {selection.get(tag.system_id, NoReferenceConfigured()) for tag in tags}
    return next(iter(selections)) if len(selections) == 1 else NoReferenceConfigured()
