"""The figures planned at each report level, and the cohort rows each one draws and stores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.figure_support import no_offline_cost
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.domain.solver.models.figure_scope import AllDatasetsScope
from neuralls.domain.solver.models.group_outcomes import (
    GroupReport,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.labels import AllMatrixSetsLabel
from neuralls.domain.solver.reporting.cohorts import (
    COHORT_COLUMN,
    MIXTURE_COHORT,
    MixtureCohort,
    RecipeCohort,
    WholeGroup,
    cohort_observations,
    cohort_rows,
)
from neuralls.platform.config.models.figure_style import FigureStyleConfig
from neuralls.platform.reporting import group_plots
from neuralls.platform.reporting.cohort_variants import is_cohort_table
from neuralls.platform.reporting.figure_card import (
    build_figure_card,
    card_payload,
    decode_figure_card,
)
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_metadata import (
    FigureKind,
    FigurePlan,
    GroupFigureMetadata,
    SiblingTable,
    TableKind,
)
from neuralls.platform.reporting.figure_planning import figure_plans, table_of
from neuralls.platform.reporting.figure_recipes import group_recipe, group_recipes
from neuralls.platform.reporting.figure_summaries import figure_summaries
from neuralls.platform.reporting.figure_text import MIXTURE_NOTE
from neuralls.platform.reporting.group_drawers import FigureData
from neuralls.platform.reporting.group_plots import GroupStyler, write_outcome_plots
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.scientific_description.policy import NotationPolicy

KEY = "key"

_GAUSSIAN_SLUG = "gaussian_m0.0_s1.0"
_SINE_SLUG = "sinusoidal_w6.28319"

_AGGREGATE_BARS = (
    "solve_time_bars",
    "online_time_bars",
    "cost_breakdown",
)
"""Pooled single figures at every level."""

_POOLED_FIGURES = (
    "paired_time_scatter",
    "speed_ratio_cdf",
    "performance_profile_solve",
    "performance_profile_online",
    "ecdf_iterations",
    "ecdf_time_solve",
    "ecdf_time_online",
)
"""Figures with one unsuffixed figure at recipe level and one pooled figure at mixture levels."""

_RECIPE_LEVEL_ONLY = (
    "breakeven",
    "matrix_heatmap_iterations",
)
"""Figures drawn only for a single recipe of a matrix set."""

_COHORT_FIGURES = (
    "convergence_factor",
    "iteration_means",
    "time_means_solve_own",
    "time_means_solve_common",
    "time_means_online_own",
    "time_means_online_common",
    "time_means_standalone_own",
    "time_means_standalone_common",
    "setup_means",
    "time_per_iteration",
    "iteration_distribution",
    "time_distribution_solve",
    "time_distribution_online",
)
"""Figures drawn once per cohort."""


def _stems(*names: str, suffix: str = "") -> set[str]:
    return {f"{name}_{KEY}{suffix}" for name in names}


@pytest.fixture
def presentation() -> GroupPresentation:
    return GroupPresentation(scope=AllDatasetsScope(), plot_styles={}, methods={})


@pytest.fixture
def all_sets_report(two_recipe_mixture_report: WorkloadMixtureReport) -> WorkloadMixtureReport:
    """The two-recipe workload pooled over every matrix set."""
    return replace(two_recipe_mixture_report, label=AllMatrixSetsLabel((GAUSSIAN, SINE)))


def _planned(report: GroupReport, presentation: GroupPresentation) -> list[FigurePlan]:
    return list(figure_plans(report, presentation, KEY, {}))


def _stems_of(plans: list[FigurePlan]) -> set[str]:
    return {plan.stem for plan in plans}


def test_a_matrix_set_and_rhs_type_report_has_one_unsuffixed_figure_of_every_kind(
    profile_report: RecipeProfileReport, presentation: GroupPresentation
) -> None:
    assert _stems_of(_planned(profile_report, presentation)) == _stems(
        *_AGGREGATE_BARS, *_POOLED_FIGURES, *_RECIPE_LEVEL_ONLY, *_COHORT_FIGURES
    )


def test_a_matrix_set_report_has_only_the_pooled_figure_of_each_cohort_kind(
    two_recipe_mixture_report: WorkloadMixtureReport, presentation: GroupPresentation
) -> None:
    assert _stems_of(_planned(two_recipe_mixture_report, presentation)) == (
        _stems(*_AGGREGATE_BARS)
        | _stems(*_POOLED_FIGURES, suffix="__mixture")
        | _stems(*_COHORT_FIGURES, suffix="__mixture")
    )


def test_an_all_matrix_sets_report_has_a_figure_per_recipe_and_the_pooled_one(
    all_sets_report: WorkloadMixtureReport, presentation: GroupPresentation
) -> None:
    assert _stems_of(_planned(all_sets_report, presentation)) == (
        _stems(*_AGGREGATE_BARS)
        | _stems(*_POOLED_FIGURES, suffix="__mixture")
        | _stems(*_COHORT_FIGURES, suffix=f"__{_GAUSSIAN_SLUG}")
        | _stems(*_COHORT_FIGURES, suffix=f"__{_SINE_SLUG}")
        | _stems(*_COHORT_FIGURES, suffix="__mixture")
    )


def test_a_cohort_figure_records_its_cohort_and_the_evidence_of_its_systems(
    all_sets_report: WorkloadMixtureReport, presentation: GroupPresentation
) -> None:
    plans = {plan.stem: plan.metadata for plan in _planned(all_sets_report, presentation)}

    sine = plans[f"iteration_means_{KEY}__{_SINE_SLUG}"]
    pooled = plans[f"iteration_means_{KEY}__mixture"]
    bars = plans[f"solve_time_bars_{KEY}"]

    assert sine.cohort == RecipeCohort(SINE)
    assert (sine.recipes, sine.recipe_counts, sine.recipe_weights) == (
        (SINE,),
        (all_sets_report.recipe_counts[1],),
        (1.0,),
    )
    assert sine.n_planned_rhs == all_sets_report.recipe_counts[1]
    assert pooled.cohort == MixtureCohort()
    assert pooled.recipes == (GAUSSIAN, SINE)
    assert pooled.n_planned_rhs == all_sets_report.n_planned_rhs
    assert bars.cohort == WholeGroup()


def test_every_figure_of_a_report_has_its_own_sibling_table_named_by_its_stem(
    all_sets_report: WorkloadMixtureReport, presentation: GroupPresentation
) -> None:
    for plan in _planned(all_sets_report, presentation):
        metadata = plan.metadata
        recipe = group_recipe(metadata.kind)
        assert table_of(plan.stem, metadata.kind) == SiblingTable(
            f"{plan.stem}.parquet", recipe.table
        )


def test_the_variant_of_a_suffixed_figure_is_found_from_its_metadata(
    all_sets_report: WorkloadMixtureReport, presentation: GroupPresentation
) -> None:
    plans = _planned(all_sets_report, presentation)

    found = {
        plan.stem: group_recipe(plan.metadata.kind)
        .variant(plan.metadata.metric, plan.metadata.population)
        .stem
        for plan in plans
    }

    assert found[f"time_means_online_common_{KEY}__{_SINE_SLUG}"] == "time_means_online_common"
    assert found[f"time_distribution_solve_{KEY}__mixture"] == "time_distribution_solve"


@pytest.mark.parametrize("kind", [FigureKind.ITERATION_MEANS, FigureKind.TIME_PER_ITERATION])
def test_the_cohort_tables_split_the_full_table_without_losing_a_row(
    all_sets_report: WorkloadMixtureReport, kind: FigureKind
) -> None:
    recipe = group_recipe(kind)
    stem = recipe.variants[0].stem
    table = figure_summaries(all_sets_report, no_offline_cost(all_sets_report))[stem]

    parts = [
        cohort_rows(table, cohort)
        for cohort in (RecipeCohort(GAUSSIAN), RecipeCohort(SINE), MixtureCohort())
    ]

    assert [set(part[COHORT_COLUMN]) for part in parts] == [
        {recipe_key(GAUSSIAN)},
        {recipe_key(SINE)},
        {MIXTURE_COHORT},
    ]
    assert_frame_equal(pl.concat(parts), table)


@dataclass(frozen=True, slots=True)
class Drawn:
    """What one figure's drawer was handed."""

    stem: str
    data: FigureData


@pytest.fixture
def drawn(monkeypatch: pytest.MonkeyPatch) -> list[Drawn]:
    """Replace every recipe's drawer by a recorder, so no figure is rendered."""
    calls: list[Drawn] = []

    def recording_recipe(kind: FigureKind):
        def draw(data: FigureData, target: FigureTarget) -> FigureTarget:
            calls.append(Drawn(target.stem, data))
            return target

        return replace(group_recipe(kind), draw=draw)

    monkeypatch.setattr(group_plots, "group_recipe", recording_recipe)
    return calls


@pytest.fixture
def write_all(
    all_sets_report: WorkloadMixtureReport,
    presentation: GroupPresentation,
    notation: NotationPolicy,
    tmp_path: Path,
    drawn: list[Drawn],
) -> Callable[[], tuple[dict[str, GroupFigureMetadata], dict[str, pl.DataFrame]]]:
    def write() -> tuple[dict[str, GroupFigureMetadata], dict[str, pl.DataFrame]]:
        plans = figure_plans(all_sets_report, presentation, KEY, {})
        staged = write_outcome_plots(
            all_sets_report,
            figure_summaries(all_sets_report, no_offline_cost(all_sets_report)),
            GroupStyler(presentation, notation, FigureStyleConfig()),
            tmp_path,
            plans,
        )
        return (
            {plan.stem: plan.metadata for plan in plans},
            {figure.figure.plan.stem: figure.table for figure in staged},
        )

    return write


def test_a_cohort_figure_draws_and_stores_the_rows_of_its_cohort(
    all_sets_report: WorkloadMixtureReport,
    drawn: list[Drawn],
    write_all: Callable[[], tuple[dict[str, GroupFigureMetadata], dict[str, pl.DataFrame]]],
) -> None:
    metadata, tables = write_all()
    full = figure_summaries(all_sets_report, no_offline_cost(all_sets_report))

    assert set(tables) == set(metadata)
    for stem, table in tables.items():
        plan = metadata[stem]
        variant = group_recipe(plan.kind).variant(plan.metric, plan.population).stem
        expected = full[variant]
        if is_cohort_table(group_recipe(plan.kind).table):
            expected = cohort_rows(expected, plan.cohort)
        assert_frame_equal(table, expected)
    cohort_stem = f"iteration_means_{KEY}__{_SINE_SLUG}"
    (call,) = (call for call in drawn if call.stem == cohort_stem)
    assert_frame_equal(call.data.summary, tables[cohort_stem])
    assert_frame_equal(
        call.data.observations,
        cohort_observations(all_sets_report.observations, RecipeCohort(SINE)),
    )


def test_the_pooled_figure_draws_every_system(
    all_sets_report: WorkloadMixtureReport,
    drawn: list[Drawn],
    write_all: Callable[[], tuple[dict[str, GroupFigureMetadata], dict[str, pl.DataFrame]]],
) -> None:
    write_all()

    (call,) = (call for call in drawn if call.stem == f"iteration_means_{KEY}__mixture")
    assert_frame_equal(call.data.observations, all_sets_report.observations)


def test_a_cohort_figure_names_its_recipe_and_the_pooled_figure_the_mixture(
    all_sets_report: WorkloadMixtureReport,
    drawn: list[Drawn],
    write_all: Callable[[], tuple[dict[str, GroupFigureMetadata], dict[str, pl.DataFrame]]],
) -> None:
    write_all()

    subtitles = {call.stem: call.data.style.subtitle for call in drawn}
    assert (
        subtitles[f"iteration_means_{KEY}__{_SINE_SLUG}"]
        .splitlines()[-1]
        .startswith("RHS recipe: ")
    )
    assert (
        subtitles[f"iteration_means_{KEY}__mixture"]
        .splitlines()[-1]
        .startswith("Mixture of recipes")
    )
    assert len(subtitles[f"solve_time_bars_{KEY}"].splitlines()) == 1


@pytest.mark.parametrize(
    "stem", ["ecdf_iterations", "performance_profile_solve", "speed_ratio_cdf"]
)
def test_a_pooled_profile_figure_is_a_mixture_figure_drawn_whole(
    all_sets_report: WorkloadMixtureReport,
    drawn: list[Drawn],
    write_all: Callable[[], tuple[dict[str, GroupFigureMetadata], dict[str, pl.DataFrame]]],
    stem: str,
) -> None:
    metadata, tables = write_all()
    full = figure_summaries(all_sets_report, no_offline_cost(all_sets_report))
    pooled = f"{stem}_{KEY}__mixture"

    (call,) = (call for call in drawn if call.stem == pooled)

    assert metadata[pooled].cohort == MixtureCohort()
    assert_frame_equal(tables[pooled], full[stem])
    assert_frame_equal(call.data.observations, all_sets_report.observations)
    assert MIXTURE_NOTE in call.data.style.subtitle.splitlines()


def test_the_kinds_of_the_per_cohort_figures_are_exactly_the_cohort_table_kinds() -> None:
    per_cohort = {recipe.kind.value for recipe in group_recipes() if is_cohort_table(recipe.table)}
    assert per_cohort == {
        "convergence_factor",
        "iteration_means",
        "time_means",
        "setup_means",
        "time_per_iteration",
        "iteration_distribution",
        "time_distribution",
    }
    assert {recipe.table for recipe in group_recipes() if recipe.kind.value in per_cohort} == {
        TableKind.COHORT_SUMMARY,
        TableKind.COHORT_RATIO_OF_SUMS,
    }


@pytest.mark.parametrize("suffix", [f"__{_SINE_SLUG}", "__mixture"])
def test_a_cohort_figures_card_round_trips_its_cohort_and_names_its_own_table(
    all_sets_report: WorkloadMixtureReport,
    presentation: GroupPresentation,
    notation: NotationPolicy,
    suffix: str,
) -> None:
    (plan,) = (
        plan
        for plan in _planned(all_sets_report, presentation)
        if plan.stem == f"iteration_means_{KEY}{suffix}"
    )

    card = build_figure_card(
        plan.stem, plan.title, plan.metadata, session_run_id="session", notation=notation
    )
    decoded = decode_figure_card(card_payload(card))

    assert decoded == card
    assert decoded.metadata == plan.metadata
    assert decoded.table == SiblingTable(f"{plan.stem}.parquet", TableKind.COHORT_SUMMARY)
