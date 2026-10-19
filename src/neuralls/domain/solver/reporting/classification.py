"""Reduce one report level to a closed outcome per report group."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from neuralls.domain.benchmarking.reference_comparison import compare_to_reference
from neuralls.domain.solver.models.benchmark_events import SystemTag
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.group_outcomes import (
    ComparisonGroupOutcome,
    GroupAggregated,
    GroupSkippedAllCacheHit,
)
from neuralls.domain.solver.models.labels import render_group_id
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
from neuralls.domain.solver.reporting.aggregation import (
    convergence_by_level,
    outcomes_by_system,
    report_level,
)
from neuralls.domain.solver.reporting.metrics_frame import build_metrics_frame
from neuralls.domain.solver.reporting.report_levels import (
    ReportLevel,
    label_for_tags,
    label_key,
    report_key,
    system_counts,
)
from neuralls.shared.constants import DEFAULT_RANDOM_SEED


def classify_level(
    planned: Sequence[SystemTag],
    results_by_system: Mapping[SystemTag, Mapping[str, CGComparisonResult]],
    sharing_by_name: Mapping[str, SetupSharingPolicy],
    level: ReportLevel,
) -> tuple[ComparisonGroupOutcome, ...]:
    """Reduce one report level to a closed outcome per report group.

    Pure: ``planned`` names every system of the case, ``results_by_system``
    only those that produced results this run (cache hits and failures carry
    none). A group with at least one measured system is aggregated; one with
    none is `GroupSkippedAllCacheHit`. A group of a single system is still
    reported (without a spread); the only statistical control is the
    per-system timing repeat policy.

    Args:
        planned: Tag of every planned system, in plan order.
        results_by_system: Results of each system that produced some this run.
        sharing_by_name: Each preconditioner name's `SetupSharingPolicy`.
        level: Which columns, besides the preconditioner, key the groups.

    Returns:
        One outcome per group of ``planned`` at ``level``, in first-seen order.

    Raises:
        KeyError: If a measured preconditioner has no entry in ``sharing_by_name``.
    """
    frame = build_metrics_frame(results_by_system, sharing_by_name)
    projected_by_label = report_level(frame, level, sharing_by_name) if frame.height else {}
    convergence_by_label = (
        convergence_by_level(frame, level, sharing_by_name) if frame.height else {}
    )
    outcomes_by_label = outcomes_by_system(frame, level) if frame.height else {}
    counts_by_label = system_counts(frame, level) if frame.height else {}
    measured = {label_key(label): label for label in counts_by_label}

    tags_by_key: dict[tuple[str, ...], list[SystemTag]] = {}
    for tag in planned:
        tags_by_key.setdefault(report_key(level, tag), []).append(tag)

    outcomes: list[ComparisonGroupOutcome] = []
    for key, tags in tags_by_key.items():
        label = measured.get(key)
        if label is None:
            skipped_label = label_for_tags(level, tags)
            outcomes.append(
                GroupSkippedAllCacheHit(
                    group_id=render_group_id(skipped_label),
                    label=skipped_label,
                    n_in_group=len(tags),
                )
            )
            continue
        counts = counts_by_label[label]
        outcomes.append(
            GroupAggregated(
                group_id=render_group_id(label),
                label=label,
                projected=projected_by_label.get(label, {}),
                n_aggregated=counts.n_systems,
                n_matrices=counts.n_matrices,
                n_in_group=len(tags),
                convergence=convergence_by_label[label],
                outcomes=outcomes_by_label[label],
                comparisons=compare_to_reference(
                    outcomes_by_label[label],
                    projected_by_label.get(label, {}),
                    seed=DEFAULT_RANDOM_SEED,
                ),
            )
        )
    return tuple(outcomes)
