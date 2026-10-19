"""Outcome of aggregating one report group's results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.labels import GroupLabel
from neuralls.domain.solver.models.outcomes import SystemOutcome
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.statistics import MethodComparison
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
        convergence: For every preconditioner name run in the group, including
            those absent from `projected`, its success count and ERT.
        outcomes: For each preconditioner name, every system's `SystemOutcome`
            keyed by system id — the individual points behind the projection's
            means, and the failures the profiles and scatter need.
        comparisons: For each non-reference preconditioner name, its break-even
            and paired comparison against the group's reference method.
    """

    group_id: GroupId
    label: GroupLabel
    projected: Mapping[str, ProjectedComparisonResult]
    n_aggregated: int
    n_matrices: int
    n_in_group: int
    convergence: Mapping[str, ConvergenceSummary] = field(default_factory=dict)
    outcomes: Mapping[str, Mapping[str, SystemOutcome]] = field(default_factory=dict)
    comparisons: Mapping[str, MethodComparison] = field(default_factory=dict)


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
