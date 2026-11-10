"""Which cohort figures a report level gets, and the report each recipe cohort is drawn from.

A figure that summarizes per recipe and pooled (a cohort table) is drawn once per cohort, each on
its own axes, instead of side by side. The level decides which cohorts get a figure:

- matrix set and RHS type: the report has one recipe, so one figure draws it, with no suffix;
- matrix set: a recipe cohort would repeat the matrix-set-and-RHS-type report of that recipe
  number for number, so only the pooled figure is drawn;
- all matrix sets: one figure per recipe and the pooled one, since no other report has them.

Figures over profiles, paired systems and speed ratios have no per-recipe rows to split; at the
two mixture levels they are one figure pooled over every system of the group, and at recipe level
the one figure of the group. Figures whose table is neither draw the whole group once at every
level.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import assert_never

import polars as pl

from neuralls.domain.solver.models.group_outcomes import GroupReport
from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    GroupLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
)
from neuralls.domain.solver.reporting.cohorts import (
    Cohort,
    MixtureCohort,
    RecipeCohort,
    WholeGroup,
    cohort_observations,
)
from neuralls.platform.reporting.figure_metadata import TableKind, text_slug
from neuralls.shared.rhs_recipe import RhsRecipe, recipe_key

MIXTURE_STEM_SUFFIX = "__mixture"
"""Ends the stem of the figure pooled over every recipe of a mixture."""

_RECIPE_STEM_PREFIX = "__"
"""Starts the recipe slug at the end of a recipe cohort's stem; two underscores never occur in the
slug of a group, so a cohort suffix cannot be read as part of the group's own name."""


class CohortLevel(StrEnum):
    """How many recipes a report group pools, which decides what its cohort figures are."""

    RECIPE = "recipe"
    MATRIX_SET = "matrix_set"
    ALL_MATRIX_SETS = "all_matrix_sets"


def cohort_level(label: GroupLabel) -> CohortLevel:
    """The cohort level of the report group named by ``label``."""
    match label:
        case MatrixSetRhsTypeLabel():
            return CohortLevel.RECIPE
        case MatrixSetLabel():
            return CohortLevel.MATRIX_SET
        case AllMatrixSetsLabel():
            return CohortLevel.ALL_MATRIX_SETS
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class CohortVariant:
    """One figure of a cohort kind: the cohort it draws and the end of its file stem."""

    cohort: Cohort
    stem_suffix: str


WHOLE_GROUP = CohortVariant(WholeGroup(), "")
"""The one figure of a kind that is not split by cohort."""

POOLED = CohortVariant(MixtureCohort(), MIXTURE_STEM_SUFFIX)


def recipe_variant(recipe: RhsRecipe) -> CohortVariant:
    """The figure of ``recipe``'s own cohort, named by the slug of its lossless key."""
    return CohortVariant(
        RecipeCohort(recipe), f"{_RECIPE_STEM_PREFIX}{text_slug(recipe_key(recipe))}"
    )


class CohortSplit(StrEnum):
    """How the figures of a table are split by cohort."""

    PER_COHORT = "per_cohort"
    """One figure per recipe and the pooled one, from a table with a row block per cohort."""
    POOLED_AT_MIXTURE = "pooled_at_mixture"
    """One figure over all systems of a mixture, from a table computed on the pooled systems."""
    WHOLE_GROUP = "whole_group"
    """One figure of the whole group at every level."""


def cohort_split(table: TableKind) -> CohortSplit:
    """How figures with a ``table`` are split by cohort."""
    match table:
        case TableKind.COHORT_SUMMARY | TableKind.COHORT_RATIO_OF_SUMS:
            return CohortSplit.PER_COHORT
        case TableKind.SERIES | TableKind.PAIRED | TableKind.SPEED_RATIO:
            return CohortSplit.POOLED_AT_MIXTURE
        case (
            TableKind.SUMMARY
            | TableKind.WORKLOAD_TOTALS
            | TableKind.COST_CURVES
            | TableKind.MATRIX_BREAKDOWN
            | TableKind.COST_BREAKDOWN
        ):
            return CohortSplit.WHOLE_GROUP
        case unreachable:
            assert_never(unreachable)


def is_cohort_table(table: TableKind) -> bool:
    """Whether a ``table`` holds one row block per cohort, which each cohort figure selects."""
    return cohort_split(table) is CohortSplit.PER_COHORT


def cohort_variants(
    table: TableKind, level: CohortLevel, recipes: Sequence[RhsRecipe]
) -> tuple[CohortVariant, ...]:
    """The figures a kind with ``table`` gets at ``level`` for a group pooling ``recipes``."""
    match cohort_split(table), level:
        case CohortSplit.WHOLE_GROUP, _:
            return (WHOLE_GROUP,)
        case CohortSplit.POOLED_AT_MIXTURE, CohortLevel.RECIPE:
            return (WHOLE_GROUP,)
        case CohortSplit.POOLED_AT_MIXTURE, CohortLevel.MATRIX_SET | CohortLevel.ALL_MATRIX_SETS:
            return (POOLED,)
        case CohortSplit.PER_COHORT, CohortLevel.RECIPE:
            return (WHOLE_GROUP,)
        case CohortSplit.PER_COHORT, CohortLevel.MATRIX_SET:
            return (POOLED,)
        case CohortSplit.PER_COHORT, CohortLevel.ALL_MATRIX_SETS:
            return (*(recipe_variant(recipe) for recipe in recipes), POOLED)
        case unreachable:
            assert_never(unreachable)


def cohort_report(report: GroupReport, cohort: Cohort) -> GroupReport:
    """The report of the systems ``cohort`` draws, for the evidence its figure's card records.

    A recipe cohort keeps only its recipe's observations, planned count and matrices; the pooled
    and whole-group cohorts are the report itself.
    """
    match cohort:
        case WholeGroup() | MixtureCohort():
            return report
        case RecipeCohort(recipe=recipe):
            observations = cohort_observations(report.observations, cohort)
            position = report.recipe_membership.index(recipe)
            return replace(
                report,
                observations=observations,
                n_matrices=observations.select(pl.struct("dataset", "matrix_id").n_unique()).item(),
                n_planned_rhs=report.recipe_counts[position],
                recipe_membership=(recipe,),
                recipe_counts=(report.recipe_counts[position],),
            )
        case unreachable:
            assert_never(unreachable)
