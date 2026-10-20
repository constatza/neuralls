"""Reduce canonical benchmark events to one closed outcome per report group."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from neuralls.domain.benchmarking.reference_comparison import compare_to_reference
from neuralls.domain.solver.models.benchmark_events import MatrixSetupEvent, SolveAttempt, SystemTag
from neuralls.domain.solver.models.group_outcomes import (
    ComparisonGroupOutcome,
    GroupAggregated,
    GroupSkippedAllCacheHit,
)
from neuralls.domain.solver.models.labels import render_group_id
from neuralls.domain.solver.reporting.aggregation import convergence_by_level, outcomes_by_system
from neuralls.domain.solver.reporting.report_levels import ReportLevel, label_for_tags, report_key
from neuralls.shared.constants import DEFAULT_RANDOM_SEED


def classify_level(
    planned: Sequence[SystemTag],
    setup_events: Collection[MatrixSetupEvent],
    attempts: Collection[SolveAttempt],
    level: ReportLevel,
) -> tuple[ComparisonGroupOutcome, ...]:
    """Classify planned groups from linked setup events and solve attempts."""
    convergence = convergence_by_level(planned, setup_events, attempts, level)
    outcomes = outcomes_by_system(planned, setup_events, attempts, level)
    tags_by_key: dict[tuple[str, ...], list[SystemTag]] = {}
    for tag in planned:
        tags_by_key.setdefault(report_key(level, tag), []).append(tag)

    classified: list[ComparisonGroupOutcome] = []
    for tags in tags_by_key.values():
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
        classified.append(
            GroupAggregated(
                group_id=render_group_id(label),
                label=label,
                n_matrices=n_matrices,
                n_planned_rhs=len(tags),
                convergence=convergence_by_method,
                outcomes=group_outcomes,
                comparisons=compare_to_reference(group_outcomes, seed=DEFAULT_RANDOM_SEED),
            )
        )
    return tuple(classified)
