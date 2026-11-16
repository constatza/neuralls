"""Which cohort figures each report level gets, and the report a recipe cohort is drawn from."""

from __future__ import annotations

import pytest
from tests.figure_support import no_offline_cost
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.domain.benchmarking.paired import reference_among_methods
from neuralls.domain.benchmarking.response import NO_RESPONSE_POINTS
from neuralls.domain.solver.models.group_outcomes import WorkloadMixtureReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.labels import (
    AllMatrixSetsLabel,
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
)
from neuralls.domain.solver.reporting.cohorts import (
    COHORT_COLUMN,
    MIXTURE_COHORT,
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
from neuralls.platform.reporting.figure_planning import figure_plans
from neuralls.platform.reporting.figure_recipes import (
    compares_to_reference,
    drawn_variants,
    group_recipes,
)
from neuralls.platform.reporting.figure_summaries import figure_summaries
from neuralls.shared.rhs_recipe import recipe_key
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


def test_only_the_cohort_tables_are_drawn_per_cohort() -> None:
    assert {table for table in TableKind if is_cohort_table(table)} == {
        TableKind.COHORT_SUMMARY,
        TableKind.COHORT_RATIO_OF_SUMS,
        TableKind.RESPONSE,
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


def test_no_figure_compared_against_the_reference_is_split_by_recipe() -> None:
    """Planning resolves the reference on the whole group, so such a figure must draw it whole."""
    compared_tables = {
        recipe.table
        for recipe in group_recipes()
        if any(compares_to_reference(variant.statistic) for variant in recipe.variants)
    }

    assert [table for table in compared_tables if is_cohort_table(table)] == []


_ONE_RECIPE_MIXTURES = [
    pytest.param(MatrixSetLabel(DatasetId("ds"), (GAUSSIAN,)), id="matrix set"),
    pytest.param(AllMatrixSetsLabel((GAUSSIAN,)), id="all matrix sets"),
]


@pytest.mark.parametrize("label", _ONE_RECIPE_MIXTURES)
def test_a_mixture_level_of_one_recipe_is_drawn_as_that_recipe(
    label: MatrixSetLabel | AllMatrixSetsLabel,
) -> None:
    assert {
        table: [
            variant.stem_suffix
            for variant in cohort_variants(table, cohort_level(label), (GAUSSIAN,))
        ]
        for table in TableKind
    } == {table: [""] for table in TableKind}


def test_the_cohort_table_of_a_one_recipe_mixture_has_no_pooled_rows(
    mixture_report: WorkloadMixtureReport,
) -> None:
    table = figure_summaries(mixture_report, no_offline_cost(mixture_report))["iteration_means"]

    assert table[COHORT_COLUMN].unique().to_list() == [recipe_key(GAUSSIAN)]


def test_the_cohort_table_of_a_two_recipe_mixture_ends_with_its_pooled_rows(
    two_recipe_mixture_report: WorkloadMixtureReport,
) -> None:
    table = figure_summaries(two_recipe_mixture_report, no_offline_cost(two_recipe_mixture_report))[
        "iteration_means"
    ]

    assert table[COHORT_COLUMN].unique(maintain_order=True).to_list() == [
        recipe_key(GAUSSIAN),
        recipe_key(SINE),
        MIXTURE_COHORT,
    ]


def test_planning_a_one_recipe_mixture_gives_no_suffixed_figure(
    mixture_report: WorkloadMixtureReport, repeated_timing_presentation: GroupPresentation
) -> None:
    reference = reference_among_methods(
        mixture_report.observations, mixture_report.reference_selection
    )

    plans = figure_plans(mixture_report, repeated_timing_presentation, "key", {})

    assert [(plan.stem, plan.metadata.cohort) for plan in plans] == [
        (variant.figure_stem("key"), WholeGroup())
        for _, variant in drawn_variants(mixture_report, reference, NO_RESPONSE_POINTS)
    ]
