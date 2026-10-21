"""Outcome of aggregating one report group's results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from neuralls.domain.benchmarking.selection import NoReferenceConfigured, ReferenceSelection
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.labels import FineLabel, GlobalLabel, GroupLabel, MediumLabel
from neuralls.domain.solver.models.outcomes import SystemOutcome
from neuralls.domain.solver.models.statistics import MethodComparison
from neuralls.shared.types import GroupId, PreconditionerName, SystemId


@dataclass(frozen=True, slots=True)
class GroupAggregated:
    """One or more comparison group members contributed a real result.

    Attributes:
        group_id: Stable, unique rendering of `label`, used for filenames and
            metric keys.
        label: Which report level and keys the report belongs to; mixture
            levels name the recipes they pool.
        n_matrices: Distinct planned matrices in the group.
        n_planned_rhs: Total planned systems in the group.
        convergence: For every preconditioner name run in the group, including
            interrupted methods, its planned, attempted, and converged counts.
        outcomes: For each preconditioner name, every system's `SystemOutcome`
            keyed by system id, including censored failures.
        comparisons: Paired comparisons against the temporary observed
            reference selection retained until explicit configuration lands.
    """

    group_id: GroupId
    label: GroupLabel
    n_matrices: int
    n_planned_rhs: int
    convergence: Mapping[PreconditionerName, ConvergenceSummary] = field(default_factory=dict)
    outcomes: Mapping[PreconditionerName, Mapping[SystemId, SystemOutcome]] = field(
        default_factory=dict
    )
    comparisons: Mapping[PreconditionerName, MethodComparison] = field(default_factory=dict)
    reference_selection: ReferenceSelection = field(default_factory=NoReferenceConfigured)
    recipe_membership: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecipeProfileReport(GroupAggregated):
    """Fine-level report for one recipe; profile and ECDF data are valid here."""

    label: FineLabel

    def __post_init__(self) -> None:
        """Require exactly the recipe named by the fine-level label."""
        if self.recipe_membership != (self.label.recipe,):
            raise ValueError("a recipe profile must contain exactly its recipe")


@dataclass(frozen=True, slots=True)
class WorkloadMixtureReport(GroupAggregated):
    """Medium/global report whose recipes are explicitly named as a mixture."""

    label: MediumLabel | GlobalLabel

    def __post_init__(self) -> None:
        """Require complete, ordered, duplicate-free mixture membership."""
        if not self.recipe_membership:
            raise ValueError("a workload mixture must name every recipe it contains")
        if tuple(dict.fromkeys(self.recipe_membership)) != self.recipe_membership:
            raise ValueError("recipe membership must be ordered and unique")


@dataclass(frozen=True, slots=True)
class GroupSkippedAllCacheHit:
    """No system of this report group produced a result this run.

    Every member was a cache hit (or failed). There is no `CGComparisonResult`
    to aggregate — not the same as zero cost, just nothing measured this run.
    """

    group_id: GroupId
    label: GroupLabel
    n_planned_rhs: int
    n_matrices: int


type ComparisonGroupOutcome = RecipeProfileReport | WorkloadMixtureReport | GroupSkippedAllCacheHit
"""The outcome of attempting to aggregate one comparison group's results.

Two variants because there are exactly two cases that call for different
handling: at least one method produced a setup event, or no method produced
any benchmark event for the group.
"""
