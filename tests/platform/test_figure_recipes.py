"""The figure registry covers every kind, and each recipe is complete for its kind of figure."""

from __future__ import annotations

from dataclasses import replace

import pytest

from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.summary import Population
from neuralls.platform.reporting.figure_metadata import GROUP_FIGURE_KINDS, FigureKind
from neuralls.platform.reporting.figure_planning import (
    FIGURE_RECIPES,
    GroupFigureRecipe,
    HistoryFigureRecipe,
    NotAGroupKindError,
    NotAHistoryKindError,
    UnknownFigureVariantError,
    group_recipe,
    history_variant,
)
from neuralls.platform.reporting.method_figures import METHOD_PLOTS


def test_every_figure_kind_has_one_recipe() -> None:
    assert set(FIGURE_RECIPES) == set(FigureKind)
    assert all(recipe.kind is kind for kind, recipe in FIGURE_RECIPES.items())


def test_group_kinds_are_the_kinds_with_a_group_recipe() -> None:
    drawn_from_a_group = {
        kind for kind, recipe in FIGURE_RECIPES.items() if isinstance(recipe, GroupFigureRecipe)
    }
    assert drawn_from_a_group == GROUP_FIGURE_KINDS


def test_method_plots_are_exactly_the_kinds_with_a_history_recipe() -> None:
    histories = {
        kind for kind, recipe in FIGURE_RECIPES.items() if isinstance(recipe, HistoryFigureRecipe)
    }
    assert set(METHOD_PLOTS) == histories


def test_mixture_reports_plan_only_the_kinds_common_to_both_reports() -> None:
    mixture_kinds = {
        kind
        for kind, recipe in FIGURE_RECIPES.items()
        if isinstance(recipe, GroupFigureRecipe) and WorkloadMixtureReport in recipe.applies_to
    }
    assert mixture_kinds == {
        FigureKind.AGGREGATE_BARS,
        FigureKind.SOLVE_TIME,
        FigureKind.ONLINE_TIME,
        FigureKind.ITERATION_MEANS,
        FigureKind.TIME_MEANS,
        FigureKind.SETUP_MEANS,
        FigureKind.TIME_PER_ITERATION,
        FigureKind.ITERATION_DISTRIBUTION,
        FigureKind.TIME_DISTRIBUTION,
        FigureKind.CONVERGENCE_FACTOR,
    }
    assert all(RecipeProfileReport in group_recipe(kind).applies_to for kind in mixture_kinds)


def test_variant_lookup_rejects_a_metric_the_kind_lacks() -> None:
    with pytest.raises(UnknownFigureVariantError):
        group_recipe(FigureKind.SPEED_RATIO).variant(MetricId.ITERATIONS, Population.OWN)


def test_variant_lookup_rejects_a_population_the_kind_lacks() -> None:
    with pytest.raises(UnknownFigureVariantError):
        group_recipe(FigureKind.ITERATION_MEANS).variant(MetricId.ITERATIONS, Population.COMMON)


def test_a_variant_is_found_by_the_metric_it_summarizes() -> None:
    recipe = group_recipe(FigureKind.ECDF_TIME)

    assert [
        recipe.variant(metric, Population.OWN).stem
        for metric in (MetricId.SOLVE_S, MetricId.ONLINE_S)
    ] == ["ecdf_time_solve", "ecdf_time_online"]


def test_time_mean_variants_are_found_by_metric_and_population() -> None:
    recipe = group_recipe(FigureKind.TIME_MEANS)

    assert {
        (metric, population): recipe.variant(metric, population).stem
        for metric in (MetricId.SOLVE_S, MetricId.ONLINE_S, MetricId.STANDALONE_S)
        for population in Population
    } == {
        (MetricId.SOLVE_S, Population.OWN): "time_means_solve_own",
        (MetricId.SOLVE_S, Population.COMMON): "time_means_solve_common",
        (MetricId.ONLINE_S, Population.OWN): "time_means_online_own",
        (MetricId.ONLINE_S, Population.COMMON): "time_means_online_common",
        (MetricId.STANDALONE_S, Population.OWN): "time_means_standalone_own",
        (MetricId.STANDALONE_S, Population.COMMON): "time_means_standalone_common",
    }


def test_a_recipe_rejects_two_variants_of_one_metric_and_population() -> None:
    recipe = group_recipe(FigureKind.ITERATION_MEANS)
    (variant,) = recipe.variants

    with pytest.raises(ValueError, match="two variants"):
        replace(recipe, variants=(variant, replace(variant, stem="other")))


def test_a_recipe_accepts_variants_of_one_metric_over_two_populations() -> None:
    recipe = group_recipe(FigureKind.ITERATION_MEANS)
    (variant,) = recipe.variants

    widened = replace(
        recipe, variants=(variant, replace(variant, stem="other", population=Population.COMMON))
    )

    assert widened.variant(MetricId.ITERATIONS, Population.COMMON).stem == "other"


def test_a_group_kind_has_no_history_variant() -> None:
    with pytest.raises(NotAHistoryKindError):
        history_variant(FigureKind.ECDF_ITERATIONS)


def test_a_history_kind_has_no_group_recipe() -> None:
    with pytest.raises(NotAGroupKindError):
        group_recipe(FigureKind.RESIDUAL_HISTORY)
