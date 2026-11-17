"""Which figures each report level gets: raw-scale aggregates at one class, scale-free views pooled."""

from __future__ import annotations

import pytest
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.domain.solver.models.group_outcomes import WorkloadMixtureReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    GroupLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
)
from neuralls.domain.solver.reporting.cohorts import MixtureCohort
from neuralls.platform.reporting.cohort_variants import (
    MIXTURE_STEM_SUFFIX,
    POOLED,
    WHOLE_GROUP,
    CohortLevel,
    CohortSplit,
    CohortVariant,
    cohort_level,
    cohort_split,
    cohort_variants,
)
from neuralls.platform.reporting.figure_metadata import TableKind
from neuralls.platform.reporting.figure_planning import figure_plans
from neuralls.platform.reporting.figure_recipes import group_recipe
from neuralls.shared.types import DatasetId

_DATASET = DatasetId("ds")


@pytest.mark.parametrize(
    ("label", "level"),
    [
        (MatrixSetRhsTypeLabel(_DATASET, GAUSSIAN), CohortLevel.RECIPE),
        (MatrixSetLabel(_DATASET, (GAUSSIAN, SINE)), CohortLevel.MATRIX_SET),
        (MatrixSetLabel(_DATASET, (GAUSSIAN,)), CohortLevel.MATRIX_SET),
        (AllMatrixSetsLabel((GAUSSIAN, SINE)), CohortLevel.ALL_MATRIX_SETS),
        (AllMatrixSetsLabel((GAUSSIAN,)), CohortLevel.ALL_MATRIX_SETS),
    ],
)
def test_only_a_matrix_set_and_rhs_type_label_is_a_class(
    label: GroupLabel, level: CohortLevel
) -> None:
    assert cohort_level(label) is level


@pytest.mark.parametrize(
    ("split", "level", "expected"),
    [
        (CohortSplit.CLASS_LEVEL_ONLY, CohortLevel.RECIPE, (WHOLE_GROUP,)),
        (CohortSplit.CLASS_LEVEL_ONLY, CohortLevel.MATRIX_SET, ()),
        (CohortSplit.CLASS_LEVEL_ONLY, CohortLevel.ALL_MATRIX_SETS, ()),
        (CohortSplit.POOLED_AT_MIXTURE, CohortLevel.RECIPE, (WHOLE_GROUP,)),
        (CohortSplit.POOLED_AT_MIXTURE, CohortLevel.MATRIX_SET, (POOLED,)),
        (CohortSplit.POOLED_AT_MIXTURE, CohortLevel.ALL_MATRIX_SETS, (POOLED,)),
    ],
)
def test_a_mixture_level_draws_only_scale_free_tables_pooled(
    split: CohortSplit, level: CohortLevel, expected: tuple[CohortVariant, ...]
) -> None:
    tables = [table for table in TableKind if cohort_split(table) is split]

    assert tables
    assert {table: cohort_variants(table, level) for table in tables} == dict.fromkeys(
        tables, expected
    )


def test_only_per_pair_tables_are_scale_free() -> None:
    assert {
        table for table in TableKind if cohort_split(table) is CohortSplit.POOLED_AT_MIXTURE
    } == {TableKind.SERIES, TableKind.PAIRED, TableKind.SPEED_RATIO}


def test_planning_a_mixture_gives_only_pooled_scale_free_figures(
    two_recipe_mixture_report: WorkloadMixtureReport,
    repeated_timing_presentation: GroupPresentation,
) -> None:
    plans = figure_plans(two_recipe_mixture_report, repeated_timing_presentation, "key", {})

    assert plans
    assert {
        (
            cohort_split(group_recipe(plan.metadata.kind).table),
            plan.metadata.cohort,
            plan.stem.endswith(MIXTURE_STEM_SUFFIX),
        )
        for plan in plans
    } == {(CohortSplit.POOLED_AT_MIXTURE, MixtureCohort(), True)}
