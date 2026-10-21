"""Reduce canonical benchmark events to one closed outcome per report group."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from neuralls.domain.benchmarking.reference_comparison import compare_to_reference
from neuralls.domain.benchmarking.selection import NoReferenceConfigured, ReferenceSelection
from neuralls.domain.solver.models.benchmark_events import MatrixSetupEvent, SolveAttempt, SystemTag
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
from neuralls.domain.solver.reporting.aggregation import convergence_by_level, outcomes_by_system
from neuralls.domain.solver.reporting.report_levels import ReportLevel, label_for_tags, report_key
from neuralls.shared.constants import DEFAULT_RANDOM_SEED
from neuralls.shared.types import SystemId


def classify_level(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
    level: ReportLevel,
    selection: ReferenceSelection | Mapping[SystemId, ReferenceSelection] | None = None,
) -> tuple[ComparisonGroupOutcome, ...]:
    """Classify planned groups from linked setup events and solve attempts."""
    convergence = convergence_by_level(planned, setup_events, attempts, level)
    outcomes = outcomes_by_system(planned, setup_events, attempts, level)
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
        group_outcomes = {
            method: outcomes.get(label, {}).get(method, {}) for method in convergence_by_method
        }
        recipe_membership = tuple(dict.fromkeys(tag.recipe for tag in tags))
        comparisons = compare_to_reference(
            group_outcomes, selection=selected, seed=DEFAULT_RANDOM_SEED
        )
        match label:
            case FineLabel():
                classified.append(
                    RecipeProfileReport(
                        group_id=render_group_id(label),
                        label=label,
                        n_matrices=n_matrices,
                        n_planned_rhs=len(tags),
                        convergence=convergence_by_method,
                        outcomes=group_outcomes,
                        comparisons=comparisons,
                        reference_selection=selected,
                        recipe_membership=recipe_membership,
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
                        outcomes=group_outcomes,
                        comparisons=comparisons,
                        reference_selection=selected,
                        recipe_membership=recipe_membership,
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
