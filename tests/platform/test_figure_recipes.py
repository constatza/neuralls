"""The figure-kind registry covers every kind, and every drawer matches a planned variant."""

from __future__ import annotations

import pytest

from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.platform.reporting.figure_metadata import GROUP_FIGURE_KINDS, FigureKind
from neuralls.platform.reporting.figure_planning import (
    FIGURE_RECIPES,
    UnknownFigureVariantError,
)
from neuralls.platform.reporting.group_plots import GROUP_DRAWERS
from neuralls.platform.reporting.method_figures import METHOD_PLOTS


def test_every_figure_kind_has_one_recipe() -> None:
    assert set(FIGURE_RECIPES) == set(FigureKind)
    assert all(recipe.kind is kind for kind, recipe in FIGURE_RECIPES.items())


def test_group_drawers_are_exactly_the_group_variants() -> None:
    planned = {
        (recipe.kind, variant.scenario)
        for recipe in FIGURE_RECIPES.values()
        if recipe.applies_to
        for variant in recipe.variants
    }
    assert set(GROUP_DRAWERS) == planned


def test_group_kinds_are_the_kinds_that_apply_to_a_group_report() -> None:
    applied = {kind for kind, recipe in FIGURE_RECIPES.items() if recipe.applies_to}
    assert applied == GROUP_FIGURE_KINDS


def test_method_plots_are_exactly_the_method_kinds() -> None:
    assert set(METHOD_PLOTS) == set(FigureKind) - GROUP_FIGURE_KINDS


def test_mixture_reports_plan_only_the_kinds_common_to_both_reports() -> None:
    mixture_kinds = {
        kind
        for kind, recipe in FIGURE_RECIPES.items()
        if WorkloadMixtureReport in recipe.applies_to
    }
    assert mixture_kinds == {
        FigureKind.AGGREGATE_BARS,
        FigureKind.SOLVE_TIME,
        FigureKind.ONLINE_TIME,
    }
    assert all(RecipeProfileReport in FIGURE_RECIPES[kind].applies_to for kind in mixture_kinds)


def test_variant_lookup_rejects_a_scenario_the_kind_lacks() -> None:
    with pytest.raises(UnknownFigureVariantError):
        FIGURE_RECIPES[FigureKind.SPEED_RATIO].variant(None)
