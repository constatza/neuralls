"""Which figures a report level gets: raw-scale aggregates at one class, scale-free views pooled.

A number that pools several RHS classes, or several matrix sets, depends on the arbitrary weights
of the mixture: a mean time, a workload sum, a quantile or a break-even point of a pooled set
moves when one class gets more systems, though no method changed (Dolan & Moré 2002, *Benchmarking
optimization software with performance profiles*; Hansen et al., the COCO platform). So:

- the matrix-set-and-RHS-type level (one class, one matrix set) gets every figure, of the whole
  group, with no stem suffix;
- the matrix-set and all-matrix-sets levels are mixtures and get only the scale-free per-pair
  views (ECDFs, performance profiles, paired scatter, speed-ratio CDF), each one figure pooled over
  every system of the group, whose card names it an equal-weight mixture of its recipes: every
  pair counts once, so a class weighs what its pair count does. Their raw-scale figures are not
  drawn: the class-level reports already draw them per class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    GroupLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
)
from neuralls.domain.solver.reporting.cohorts import Cohort, MixtureCohort, WholeGroup
from neuralls.platform.reporting.figure_metadata import TableKind

MIXTURE_STEM_SUFFIX = "__mixture"
"""Ends the stem of a figure pooled over every system of a mixture level."""


class CohortLevel(StrEnum):
    """Whether a report group is one class or a mixture, which decides the figures it gets."""

    RECIPE = "recipe"
    """One matrix set and one RHS class: a matrix-set-and-RHS-type group."""
    MATRIX_SET = "matrix_set"
    ALL_MATRIX_SETS = "all_matrix_sets"


def cohort_level(label: GroupLabel) -> CohortLevel:
    """The cohort level of the report group named by ``label``.

    A matrix-set or all-matrix-sets group is a mixture even when it pools one recipe: the
    all-matrix-sets group still pools matrix sets, and a one-recipe matrix-set group repeats its
    class-level report, so neither gets a raw-scale aggregate.
    """
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
    """One figure of a kind: the cohort it draws and the end of its file stem."""

    cohort: Cohort
    stem_suffix: str


WHOLE_GROUP = CohortVariant(WholeGroup(), "")
"""The one figure of a kind at the class level."""

POOLED = CohortVariant(MixtureCohort(), MIXTURE_STEM_SUFFIX)
"""The one figure of a scale-free kind at a mixture level."""


class CohortSplit(StrEnum):
    """Whether the figures of a table may pool a mixture."""

    CLASS_LEVEL_ONLY = "class_level_only"
    """A raw-scale aggregate, whose pooled value would depend on the mixture's weights."""
    POOLED_AT_MIXTURE = "pooled_at_mixture"
    """A scale-free per-pair view, drawn once over all systems of a mixture."""


def cohort_split(table: TableKind) -> CohortSplit:
    """Whether figures with a ``table`` may pool a mixture."""
    match table:
        case TableKind.SERIES | TableKind.PAIRED | TableKind.SPEED_RATIO:
            return CohortSplit.POOLED_AT_MIXTURE
        case (
            TableKind.SUMMARY
            | TableKind.WORKLOAD
            | TableKind.COST_CURVES
            | TableKind.COHORT_SUMMARY
            | TableKind.COHORT_RATIO_OF_SUMS
            | TableKind.MATRIX_BREAKDOWN
            | TableKind.FULL_COST_BREAKDOWN
            | TableKind.BREAKEVEN_FULL
        ):
            return CohortSplit.CLASS_LEVEL_ONLY
        case unreachable:
            assert_never(unreachable)


def cohort_variants(table: TableKind, level: CohortLevel) -> tuple[CohortVariant, ...]:
    """The figures a kind with ``table`` gets at ``level``: none, or one."""
    match cohort_split(table), level:
        case _, CohortLevel.RECIPE:
            return (WHOLE_GROUP,)
        case CohortSplit.POOLED_AT_MIXTURE, CohortLevel.MATRIX_SET | CohortLevel.ALL_MATRIX_SETS:
            return (POOLED,)
        case CohortSplit.CLASS_LEVEL_ONLY, CohortLevel.MATRIX_SET | CohortLevel.ALL_MATRIX_SETS:
            return ()
        case unreachable:
            assert_never(unreachable)
