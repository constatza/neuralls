"""Outcome of aggregating one report group's results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from neuralls.domain.solver.models.labels import (
    GroupLabel,
)
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.shared.types import GroupId


@dataclass(frozen=True, slots=True)
class GroupAggregated:
    """One or more comparison group members contributed a real result.

    Attributes:
        group_id: Stable, unique rendering of `label`, used for filenames and
            metric keys.
        label: Which report level and keys the projection belongs to; mixture
            levels name the recipes they pool.
        projected: One `ProjectedComparisonResult` per preconditioner name.
        n_aggregated: Systems actually contributing a result this run.
        n_matrices: Distinct matrices those systems solve on.
        n_in_group: Total planned systems of the group; `n_aggregated < n_in_group`
            means the rest were cache hits (reused from an earlier run,
            carrying no result to aggregate this time) or failed.
        time_to_solution_by_system: For each preconditioner name, each
            contributing system's time-to-solution in seconds, keyed by
            system id in plan order — the individual points behind the
            projection's means. A system whose time is incomparable (an
            unavailable cost) is omitted for that preconditioner.
    """

    group_id: GroupId
    label: GroupLabel
    projected: Mapping[str, ProjectedComparisonResult]
    n_aggregated: int
    n_matrices: int
    n_in_group: int
    time_to_solution_by_system: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GroupSkippedAllCacheHit:
    """No system of this report group produced a result this run.

    Every member was a cache hit (or failed). There is no `CGComparisonResult`
    to aggregate — not the same as zero cost, just nothing measured this run.
    """

    group_id: GroupId
    label: GroupLabel
    n_in_group: int


type ComparisonGroupOutcome = GroupAggregated | GroupSkippedAllCacheHit
"""The outcome of attempting to aggregate one comparison group's results.

Two variants because there are exactly two cases that call for different
handling: there is a projection to act on, or there is nothing measured this
run. A group of one system is still a projection (without a spread). Whether every
member contributed (`n_aggregated == n_in_group`) or only some did is a
fact carried as data on `GroupAggregated`, not another variant — both
cases are plotted/logged/uploaded identically, differing only in whether a
warning fires.
"""
