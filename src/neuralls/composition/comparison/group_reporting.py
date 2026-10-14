"""Classify and report every discovered comparison group's aggregation outcome.

Orchestration only: which groups get a projection, what is logged when one
does not, and how setup-sharing policy is derived from a group's cache. The
destination of an actual projection is the injected `GroupReportSink`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loguru import logger

from neuralls.composition.comparison._generation_cost import resolve_generation_repeatability
from neuralls.composition.comparison._preconditioner_setup import GroupPreconditionerCache
from neuralls.composition.comparison.models import (
    ComparisonGroup,
    ComparisonGroupings,
    ComparisonOutcome,
)
from neuralls.domain.solver.comparison import classify_group_aggregation
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    GroupAggregated,
    GroupBelowMinimumSize,
    GroupSkippedAllCacheHit,
    SetupSharingPolicy,
)
from neuralls.domain.solver.ports import GroupReportSink
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.shared.constants import MIN_GROUP_SIZE
from neuralls.shared.types import GenerationRepeatability, GroupId

__all__ = ["report_all_groupings"]


def _sharing_by_name(
    specs: Sequence[PreconditionerConfig], cache: GroupPreconditionerCache | None
) -> dict[str, SetupSharingPolicy]:
    """Each preconditioner name's setup/generation sharing policy for one group."""
    shared_names = cache.shared_names() if cache is not None else frozenset[str]()
    return {
        spec.name: SetupSharingPolicy(
            share_setup=spec.name in shared_names,
            share_generation=resolve_generation_repeatability(spec)
            is GenerationRepeatability.ONE_TIME,
        )
        for spec in specs
    }


def _group_specs(
    group: ComparisonGroup, specs_by_entry_id: Mapping[str, Sequence[PreconditionerConfig]]
) -> list[PreconditionerConfig]:
    """Every distinct preconditioner definition across the group's members.

    Members may run different preconditioner *lists*; a name present in only
    some of them still needs a sharing policy. Where members share a name the
    definitions are identical (`validate_group_specs_consistent`), so the first
    occurrence in config order represents it.
    """
    by_name: dict[str, PreconditionerConfig] = {}
    for entry_id in group.entry_ids:
        for spec in specs_by_entry_id.get(entry_id, ()):
            by_name.setdefault(spec.name, spec)
    return list(by_name.values())


def _title(group: ComparisonGroup) -> str:
    """Plot title naming the group, and its RHS composition when it pools several kinds."""
    if len(group.rhs_kind_counts) <= 1:
        return f"Projected cost per system vs. N — group '{group.group_id}'"
    mixture = ", ".join(f"{kind} ×{count}" for kind, count in group.rhs_kind_counts)
    return f"Projected cost per system vs. N — pooled mixture '{group.group_id}' ({mixture})"


def _unmeasured_note(n_unmeasured: int, n_failed: int) -> str:
    """Why members produced no result: cache hits carry none, and failures never had one."""
    return f"{n_unmeasured - n_failed} cache hit(s), {n_failed} failed"


def _report_group(
    group: ComparisonGroup,
    outcomes_by_id: Mapping[str, ComparisonOutcome],
    specs_by_entry_id: Mapping[str, Sequence[PreconditionerConfig]],
    cache: GroupPreconditionerCache | None,
    sink: GroupReportSink,
) -> None:
    results_by_system: dict[str, dict[str, CGComparisonResult]] = {}
    n_failed = 0
    for entry_id in group.entry_ids:
        outcome = outcomes_by_id.get(entry_id)
        if outcome is None:
            continue
        if outcome.payload is not None:
            results_by_system[entry_id] = outcome.payload.results
        elif not outcome.success:
            n_failed += 1

    classified = classify_group_aggregation(
        group.group_id,
        len(group.entry_ids),
        results_by_system,
        _sharing_by_name(_group_specs(group, specs_by_entry_id), cache),
        min_size=MIN_GROUP_SIZE,
    )
    match classified:
        case GroupAggregated() as aggregated:
            if aggregated.n_aggregated < aggregated.n_in_group:
                logger.warning(
                    "Group '{}': aggregated {} of {} members — {}.",
                    aggregated.group_id,
                    aggregated.n_aggregated,
                    aggregated.n_in_group,
                    _unmeasured_note(aggregated.n_in_group - aggregated.n_aggregated, n_failed),
                )
            sink.report(aggregated, title=_title(group))
        case GroupSkippedAllCacheHit(group_id=group_id, n_in_group=n_total):
            logger.warning(
                "Group '{}': nothing measured this run among {} members — {}.",
                group_id,
                n_total,
                _unmeasured_note(n_total, n_failed),
            )
        case GroupBelowMinimumSize(
            group_id=group_id, n_aggregated=n_done, n_in_group=n_total, min_size=min_size
        ):
            logger.warning(
                "Group '{}': only {} of {} members produced a result, below the minimum of {} "
                "needed for a scaling projection — skipped.",
                group_id,
                n_done,
                n_total,
                min_size,
            )


def report_all_groupings(
    outcomes: Sequence[ComparisonOutcome],
    specs_by_entry_id: Mapping[str, Sequence[PreconditionerConfig]],
    groupings: ComparisonGroupings,
    group_caches: Mapping[GroupId, GroupPreconditionerCache],
    sink: GroupReportSink,
) -> None:
    """Report both grouping granularities for one batch.

    Visits every discovered group — including one where every member was a
    cache hit — so a skipped group is logged loudly instead of silently
    producing nothing. The sink is called only for a group with a
    projection, so with every group skipped no destination is touched.

    Args:
        outcomes: Every outcome the batch is about to return.
        specs_by_entry_id: Resolved preconditioner definitions of each entry
            that executed (absent for cache hits).
        groupings: Both discovered groupings.
        group_caches: Each coarse group's preconditioner cache, if one was
            constructed (empty for a fully-cache-hit batch).
        sink: Receives each aggregated group's projection.
    """
    outcomes_by_id = {outcome.comparison_id: outcome for outcome in outcomes}
    for groups in (groupings.by_matrix_and_rhs_kind, groupings.by_matrix):
        for group in groups.values():
            scope = groupings.cache_scope(group)
            cache = group_caches.get(scope.group_id) if scope is not None else None
            _report_group(group, outcomes_by_id, specs_by_entry_id, cache, sink)
