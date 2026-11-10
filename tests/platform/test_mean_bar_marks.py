"""The marks of mean and time-per-iteration bars are read from the table stored beside them.

Bars, whiskers and the "k/n" annotations come from the figure's summary frame; only the strip
points of time per iteration come from the observations. No figure is drawn.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.figure_support import no_offline_cost
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.reporting.cohorts import (
    MixtureCohort,
    NotOneCohortError,
    RecipeCohort,
    cohort_observations,
    cohort_rows,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.summary import Population, select
from neuralls.platform.reporting.cohort_points import panel_points
from neuralls.platform.reporting.figure_metadata import Statistic
from neuralls.platform.reporting.figure_summaries import figure_summaries
from neuralls.platform.reporting.seaborn_means import (
    Bar,
    BarPanel,
    MethodBar,
    NoBar,
    NoWhisker,
    Whisker,
    bar_panel,
    defined_label,
)
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.types import PreconditionerName

AMG = PreconditionerName("amg")
BROKEN = PreconditionerName("broken")
JACOBI = PreconditionerName("jacobi")


def _bars(panel: BarPanel) -> dict[str, MethodBar]:
    return {bar.method: bar for bar in panel.bars}


def _row(summary: pl.DataFrame, method: str) -> dict[str, object]:
    return summary.filter(pl.col("method") == method).row(0, named=True)


def test_a_mean_bar_and_its_whisker_are_the_summary_row_of_its_method(
    profile_report: RecipeProfileReport,
) -> None:
    summary = figure_summaries(profile_report, no_offline_cost(profile_report))["iteration_means"]

    panel = bar_panel(summary, Statistic.MEAN_OVER_SYSTEMS)

    row = _row(summary, JACOBI)
    assert _bars(panel)[JACOBI] == MethodBar(
        JACOBI,
        Bar(
            row["mean_over_systems"],
            Whisker(row["mean_over_systems_ci_low"], row["mean_over_systems_ci_high"]),
        ),
        n_defined=9,
        n_planned=9,
    )


def test_a_method_with_nothing_defined_has_no_bar_but_keeps_its_counts(
    profile_report: RecipeProfileReport,
) -> None:
    summary = figure_summaries(profile_report, no_offline_cost(profile_report))["iteration_means"]

    panel = bar_panel(summary, Statistic.MEAN_OVER_SYSTEMS)

    assert _bars(panel)[BROKEN] == MethodBar(BROKEN, NoBar(), n_defined=0, n_planned=9)


def test_setup_bars_read_the_mean_over_matrices_of_matrices_set_up(
    profile_report: RecipeProfileReport,
) -> None:
    summary = figure_summaries(profile_report, no_offline_cost(profile_report))["setup_means"]

    panel = bar_panel(summary, Statistic.MEAN_OVER_MATRICES)

    amg = _bars(panel)[AMG]
    row = _row(summary, AMG)
    assert amg.height == Bar(
        row["mean_over_matrices"],
        Whisker(row["mean_over_matrices_ci_low"], row["mean_over_matrices_ci_high"]),
    )
    # amg's setup completed on matrices 1 and 2, failed on 3 and never ran on 4
    assert (amg.n_defined, amg.n_planned) == (2, 4)


def test_time_per_iteration_bars_are_the_ratio_of_sums_without_a_whisker(
    profile_report: RecipeProfileReport,
) -> None:
    summary = figure_summaries(profile_report, no_offline_cost(profile_report))[
        "time_per_iteration"
    ]

    panel = bar_panel(summary, Statistic.RATIO_OF_SUMS)

    assert _bars(panel)[AMG].height == Bar(_row(summary, AMG)["ratio_of_sums"], NoWhisker())


def test_each_cohort_of_a_mixture_is_its_own_panel(
    two_recipe_mixture_report: WorkloadMixtureReport,
) -> None:
    summary = figure_summaries(
        two_recipe_mixture_report, no_offline_cost(two_recipe_mixture_report)
    )["iteration_means"]

    panels = [
        bar_panel(cohort_rows(summary, cohort), Statistic.MEAN_OVER_SYSTEMS)
        for cohort in (RecipeCohort(GAUSSIAN), RecipeCohort(SINE), MixtureCohort())
    ]

    assert [panel.cohort for panel in panels] == [recipe_key(GAUSSIAN), recipe_key(SINE), "mixture"]
    # amg converged on four systems of the Gaussian matrices and none of the second recipe
    assert [_bars(panel)[AMG].n_defined for panel in panels] == [4, 0, 4]


def test_a_panel_of_several_cohorts_is_rejected(
    two_recipe_mixture_report: WorkloadMixtureReport,
) -> None:
    summary = figure_summaries(
        two_recipe_mixture_report, no_offline_cost(two_recipe_mixture_report)
    )["iteration_means"]

    with pytest.raises(NotOneCohortError):
        bar_panel(summary, Statistic.MEAN_OVER_SYSTEMS)


def test_a_panel_states_the_population_of_its_rows(
    two_recipe_mixture_report: WorkloadMixtureReport,
) -> None:
    summary = figure_summaries(
        two_recipe_mixture_report, no_offline_cost(two_recipe_mixture_report)
    )["time_means_solve_common"]

    panel = bar_panel(cohort_rows(summary, MixtureCohort()), Statistic.MEAN_OVER_SYSTEMS)

    assert panel.population == Population.COMMON


def test_time_per_iteration_points_are_each_recipes_systems_with_an_iterate(
    two_recipe_mixture_report: WorkloadMixtureReport,
) -> None:
    observations = two_recipe_mixture_report.observations
    summary = figure_summaries(
        two_recipe_mixture_report, no_offline_cost(two_recipe_mixture_report)
    )["time_per_iteration"]
    cohort = RecipeCohort(GAUSSIAN)
    first = bar_panel(cohort_rows(summary, cohort), Statistic.RATIO_OF_SUMS)

    points = panel_points(
        cohort_observations(observations, cohort), first, MetricId.TIME_PER_ITERATION_S
    )

    expected = (
        select(observations, MetricId.TIME_PER_ITERATION_S, Population.OWN)
        .filter(pl.col("recipe") == recipe_key(GAUSSIAN))
        .select("method", value="time_per_iteration_s")
    )
    assert_frame_equal(points, expected)


@pytest.mark.parametrize(
    ("metric", "label"),
    [
        (MetricId.ITERATIONS, "4/9 solved"),
        (MetricId.ONLINE_S, "4/9 solved"),
        (MetricId.SETUP_S, "4/9 set up"),
        (MetricId.TIME_PER_ITERATION_S, "4/9 returned an iterate"),
    ],
)
def test_the_annotation_counts_defined_of_planned_units_in_the_metrics_words(
    metric: MetricId, label: str
) -> None:
    bar = MethodBar(AMG, NoBar(), n_defined=4, n_planned=9)

    assert defined_label(bar, metric) == label
