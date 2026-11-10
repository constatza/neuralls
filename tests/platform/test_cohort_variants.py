"""Which cohort figures each report level gets, and the report a recipe cohort is drawn from."""

from __future__ import annotations

import pytest
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.domain.solver.models.group_outcomes import WorkloadMixtureReport
from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
)
from neuralls.domain.solver.reporting.cohorts import (
    MixtureCohort,
    RecipeCohort,
    WholeGroup,
    cohort_observations,
)
from neuralls.platform.reporting.cohort_variants import (
    MIXTURE_STEM_SUFFIX,
    CohortLevel,
    CohortSplit,
    cohort_level,
    cohort_report,
    cohort_split,
    cohort_variants,
    is_cohort_table,
    recipe_variant,
)
from neuralls.platform.reporting.figure_metadata import TableKind
from neuralls.shared.types import DatasetId

_RECIPES = (GAUSSIAN, SINE)


def _suffixes(table: TableKind, level: CohortLevel) -> list[str]:
    return [variant.stem_suffix for variant in cohort_variants(table, level, _RECIPES)]


def test_each_label_has_its_cohort_level() -> None:
    dataset = DatasetId("ds")

    assert [
        cohort_level(MatrixSetRhsTypeLabel(dataset, GAUSSIAN)),
        cohort_level(MatrixSetLabel(dataset, _RECIPES)),
        cohort_level(AllMatrixSetsLabel(_RECIPES)),
    ] == [CohortLevel.RECIPE, CohortLevel.MATRIX_SET, CohortLevel.ALL_MATRIX_SETS]


def test_a_recipe_level_report_has_one_unsuffixed_figure_of_the_whole_group() -> None:
    (variant,) = cohort_variants(TableKind.COHORT_SUMMARY, CohortLevel.RECIPE, (GAUSSIAN,))

    assert (variant.cohort, variant.stem_suffix) == (WholeGroup(), "")


def test_a_matrix_set_has_only_the_pooled_figure() -> None:
    (variant,) = cohort_variants(TableKind.COHORT_SUMMARY, CohortLevel.MATRIX_SET, _RECIPES)

    assert (variant.cohort, variant.stem_suffix) == (MixtureCohort(), MIXTURE_STEM_SUFFIX)


def test_all_matrix_sets_have_one_figure_per_recipe_then_the_pooled_one() -> None:
    variants = cohort_variants(
        TableKind.COHORT_RATIO_OF_SUMS, CohortLevel.ALL_MATRIX_SETS, _RECIPES
    )

    assert [variant.cohort for variant in variants] == [
        RecipeCohort(GAUSSIAN),
        RecipeCohort(SINE),
        MixtureCohort(),
    ]
    assert [variant.stem_suffix for variant in variants] == [
        recipe_variant(GAUSSIAN).stem_suffix,
        recipe_variant(SINE).stem_suffix,
        "__mixture",
    ]
    assert len({variant.stem_suffix for variant in variants}) == len(variants)


@pytest.mark.parametrize("level", list(CohortLevel))
@pytest.mark.parametrize(
    "table",
    [table for table in TableKind if cohort_split(table) is CohortSplit.WHOLE_GROUP],
)
def test_a_figure_without_a_cohort_table_is_one_figure_of_the_whole_group_at_every_level(
    table: TableKind, level: CohortLevel
) -> None:
    assert _suffixes(table, level) == [""]


@pytest.mark.parametrize("table", [TableKind.SERIES, TableKind.PAIRED, TableKind.SPEED_RATIO])
def test_a_profile_table_is_one_figure_at_recipe_level_and_one_pooled_figure_at_mixture_levels(
    table: TableKind,
) -> None:
    assert {level: _suffixes(table, level) for level in CohortLevel} == {
        CohortLevel.RECIPE: [""],
        CohortLevel.MATRIX_SET: ["__mixture"],
        CohortLevel.ALL_MATRIX_SETS: ["__mixture"],
    }


def test_only_the_two_cohort_tables_are_drawn_per_cohort() -> None:
    assert {table for table in TableKind if is_cohort_table(table)} == {
        TableKind.COHORT_SUMMARY,
        TableKind.COHORT_RATIO_OF_SUMS,
    }


def test_a_recipe_cohort_report_holds_only_its_recipes_systems(
    two_recipe_mixture_report: WorkloadMixtureReport,
) -> None:
    cohort = RecipeCohort(SINE)

    report = cohort_report(two_recipe_mixture_report, cohort)

    assert report.recipe_membership == (SINE,)
    assert report.recipe_counts == (two_recipe_mixture_report.recipe_counts[1],)
    assert report.n_planned_rhs == two_recipe_mixture_report.recipe_counts[1]
    assert report.observations.equals(
        cohort_observations(two_recipe_mixture_report.observations, cohort)
    )
    assert set(report.observations["recipe"]) == {"sinusoidal_w6.28319"}
    assert report.n_matrices == 2


@pytest.mark.parametrize("cohort", [WholeGroup(), MixtureCohort()])
def test_the_whole_and_pooled_cohorts_are_the_report_itself(
    two_recipe_mixture_report: WorkloadMixtureReport, cohort: WholeGroup | MixtureCohort
) -> None:
    assert cohort_report(two_recipe_mixture_report, cohort) is two_recipe_mixture_report
