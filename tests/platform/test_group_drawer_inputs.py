"""The frames and series each group drawer hands to its plot function; no figure is drawn."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.figure_support import no_offline_cost
from tests.observation_support import solved_row, synthetic_observations, unconverged_row
from tests.recipe_support import GAUSSIAN

from neuralls.domain.benchmarking.paired import PairKind
from neuralls.domain.benchmarking.profiles import ProfileSeries
from neuralls.domain.benchmarking.selection import ConfiguredReference, NoReferenceConfigured
from neuralls.domain.solver.models.group_outcomes import (
    GroupAggregated,
    RecipeProfileReport,
)
from neuralls.domain.solver.models.labels import (
    MatrixSetRhsTypeLabel,
    render_group_id,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.summary import Population
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_metadata import TABLE_SCHEMAS, FigureKind, Statistic
from neuralls.platform.reporting.figure_planning import figure_summaries, group_recipe
from neuralls.platform.reporting.group_drawers import FigureData
from neuralls.platform.reporting.outcome_plots import GroupStyle, WorkloadTotal
from neuralls.shared.types import DatasetId, PreconditionerName

AMG = PreconditionerName("amg")
JACOBI = PreconditionerName("jacobi")
TARGET = FigureTarget(Path("figures"), "figure")


@dataclass(frozen=True, slots=True)
class PlotCall:
    """The positional arguments and options one plot function was called with."""

    args: tuple[object, ...]
    options: Mapping[str, object]


@dataclass(slots=True)
class PlotCalls:
    """The call of each plot function, by plot function name."""

    by_function: dict[str, PlotCall] = field(default_factory=dict)


@pytest.fixture
def plot_calls(monkeypatch: pytest.MonkeyPatch) -> PlotCalls:
    """Replace every plot function with a recorder that returns the target it was given."""
    calls = PlotCalls()

    def recorder(name: str) -> Callable[..., FigureTarget]:
        def record(*args: object, target: FigureTarget, **kwargs: object) -> FigureTarget:
            calls.by_function[name] = PlotCall(args, kwargs)
            return target

        return record

    for name in (
        "plot_aggregate_bars",
        "plot_workload_bars",
        "plot_cost_breakdown",
        "plot_ecdf_profile",
        "plot_performance_profile",
        "plot_paired_time_scatter",
        "plot_speed_ratio_cdf",
        "plot_break_even_curves",
        "plot_convergence_factors",
        "plot_mean_bars",
        "plot_time_per_iteration",
        "plot_distribution",
        "plot_matrix_heatmap",
    ):
        monkeypatch.setattr(f"neuralls.platform.reporting.group_drawers.{name}", recorder(name))
    return calls


@pytest.fixture
def observations() -> pl.DataFrame:
    """``amg`` solves s0 and s1, ``jacobi`` s0 and s2; s0 and s1 sit on matrix 1, s2 on matrix 2.

    The online time of a solved system is its solve time plus 0.5.
    """
    return synthetic_observations(
        [
            solved_row(AMG, "s0", 1, 1.0, 3, online_seconds=1.5),
            solved_row(AMG, "s1", 1, 2.0, 4, online_seconds=2.5),
            unconverged_row(AMG, "s2", 2),
            solved_row(JACOBI, "s0", 1, 4.0, 30, online_seconds=4.5),
            unconverged_row(JACOBI, "s1", 1),
            solved_row(JACOBI, "s2", 2, 8.0, 40, online_seconds=8.5),
        ]
    )


def _recipe_report(
    observations: pl.DataFrame, reference: ConfiguredReference | NoReferenceConfigured
) -> RecipeProfileReport:
    label = MatrixSetRhsTypeLabel(dataset=DatasetId("ds"), recipe=GAUSSIAN)
    return RecipeProfileReport(
        group_id=render_group_id(label),
        label=label,
        n_matrices=2,
        n_planned_rhs=3,
        observations=observations,
        reference_selection=reference,
        recipe_membership=(GAUSSIAN,),
    )


@pytest.fixture
def report(observations: pl.DataFrame) -> RecipeProfileReport:
    return _recipe_report(observations, ConfiguredReference(AMG))


def _data(report: GroupAggregated, summary: pl.DataFrame, metric: MetricId) -> FigureData:
    return FigureData(
        report.observations, summary, MagicMock(spec=GroupStyle), metric, report.reference_selection
    )


def _draw(report: GroupAggregated, kind: FigureKind, metric: MetricId) -> FigureTarget | None:
    recipe = group_recipe(kind)
    summary = figure_summaries(report, no_offline_cost(report))[
        recipe.variant(metric, Population.OWN).stem
    ]
    return recipe.draw(_data(report, summary, metric), TARGET)


def _x(series: ProfileSeries, method: str) -> list[float]:
    return series.frame.filter(pl.col("method") == method)["x"].to_list()


def _series_of(call: PlotCall) -> ProfileSeries:
    match call.args:
        case (ProfileSeries() as series,):
            return series
        case other:
            raise TypeError(f"expected one profile series, got {other!r}")


def _frames_of(call: PlotCall) -> tuple[pl.DataFrame, ...]:
    match call.args:
        case tuple() as args if all(isinstance(arg, pl.DataFrame) for arg in args):
            return tuple(arg for arg in args if isinstance(arg, pl.DataFrame))
        case other:
            raise TypeError(f"expected frames, got {other!r}")


def test_the_iteration_ecdf_is_drawn_from_the_iterations_of_converged_systems(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    assert _draw(report, FigureKind.ECDF_ITERATIONS, MetricId.ITERATIONS) == TARGET

    call = plot_calls.by_function["plot_ecdf_profile"]
    series = _series_of(call)
    assert (_x(series, AMG), _x(series, JACOBI)) == ([3.0, 4.0], [30.0, 40.0])
    assert call.options["xlabel"] == "Iterations"


@pytest.mark.parametrize(
    ("metric", "amg", "jacobi"),
    [
        (MetricId.SOLVE_S, [1.0, 2.0], [4.0, 8.0]),
        (MetricId.ONLINE_S, [1.5, 2.5], [4.5, 8.5]),
    ],
)
def test_the_time_ecdf_is_drawn_from_the_metrics_seconds(
    report: RecipeProfileReport,
    plot_calls: PlotCalls,
    metric: MetricId,
    amg: list[float],
    jacobi: list[float],
) -> None:
    _draw(report, FigureKind.ECDF_TIME, metric)

    series = _series_of(plot_calls.by_function["plot_ecdf_profile"])
    assert (_x(series, AMG), _x(series, JACOBI)) == (amg, jacobi)


def test_the_online_performance_profile_is_drawn_from_ratios_to_the_best_method(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    _draw(report, FigureKind.PERFORMANCE_PROFILE, MetricId.ONLINE_S)

    series = _series_of(plot_calls.by_function["plot_performance_profile"])
    # s0: amg 1.5 beats jacobi 4.5; s1 only amg; s2 only jacobi
    assert _x(series, AMG) == [1.0, 1.0]
    assert _x(series, JACOBI) == [1.0, 3.0]
    assert series.n_excluded_zero_best == 0


def test_the_paired_scatter_is_drawn_from_the_online_pairs_against_the_reference(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    _draw(report, FigureKind.PAIRED_TIME, MetricId.ONLINE_S)

    call = plot_calls.by_function["plot_paired_time_scatter"]
    (paired,) = _frames_of(call)
    assert (call.options["reference_name"], call.options["metric"]) == (AMG, MetricId.ONLINE_S)
    assert paired.select("system_id", "pair_kind", "reference_value", "method_value").rows() == [
        ("s0", PairKind.BOTH, 1.5, 4.5),
        ("s1", PairKind.REFERENCE_ONLY, 2.5, None),
        ("s2", PairKind.METHOD_ONLY, None, 8.5),
    ]


def test_the_speed_ratio_cdf_is_drawn_from_the_ratios_and_their_summary(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    _draw(report, FigureKind.SPEED_RATIO, MetricId.ONLINE_S)

    call = plot_calls.by_function["plot_speed_ratio_cdf"]
    ratios, summary = _frames_of(call)
    assert call.options["reference_name"] == AMG
    assert ratios.select("method", "system_id", "ratio").rows() == [(JACOBI, "s0", 1.5 / 4.5)]
    row = summary.row(0, named=True)
    assert (row["method"], row["n_ratio"], row["n_matrices"]) == (JACOBI, 1, 1)
    assert (row["n_reference_only"], row["n_method_only"]) == (1, 1)


def test_the_break_even_curves_are_drawn_from_the_cost_curves_they_store_against_the_reference(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    summary = figure_summaries(report, no_offline_cost(report))["breakeven"]
    group_recipe(FigureKind.BREAKEVEN).draw(_data(report, summary, MetricId.STANDALONE_S), TARGET)

    call = plot_calls.by_function["plot_break_even_curves"]
    (curves,) = _frames_of(call)
    assert call.options["reference_name"] == AMG
    assert_frame_equal(curves, figure_summaries(report, no_offline_cost(report))["breakeven"])


def test_the_convergence_factors_are_drawn_from_their_cohort_summary_and_the_observations(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    metric = MetricId.CONVERGENCE_FACTOR
    _draw(report, FigureKind.CONVERGENCE_FACTOR, metric)

    call = plot_calls.by_function["plot_convergence_factors"]
    summary, observations = _frames_of(call)
    assert_frame_equal(
        summary, figure_summaries(report, no_offline_cost(report))["convergence_factor"]
    )
    assert_frame_equal(observations, report.observations)
    assert call.options["metric"] is metric


@pytest.mark.parametrize(
    ("kind", "metric"),
    [
        (FigureKind.PAIRED_TIME, MetricId.ONLINE_S),
        (FigureKind.SPEED_RATIO, MetricId.ONLINE_S),
        (FigureKind.BREAKEVEN, MetricId.STANDALONE_S),
    ],
)
def test_reference_figures_need_a_configured_reference_among_the_methods(
    observations: pl.DataFrame, plot_calls: PlotCalls, kind: FigureKind, metric: MetricId
) -> None:
    recipe = group_recipe(kind)
    empty = pl.DataFrame(schema=TABLE_SCHEMAS[recipe.table])
    unconfigured = _recipe_report(observations, NoReferenceConfigured())
    elsewhere = _recipe_report(observations, ConfiguredReference(PreconditionerName("elsewhere")))

    assert recipe.draw(_data(unconfigured, empty, metric), TARGET) is None
    assert recipe.draw(_data(elsewhere, empty, metric), TARGET) is None
    assert plot_calls.by_function == {}


@pytest.mark.parametrize(
    ("kind", "metric", "function"),
    [
        (FigureKind.AGGREGATE_BARS, MetricId.ITERATIONS, "plot_aggregate_bars"),
        (FigureKind.SOLVE_TIME, MetricId.SOLVE_S, "plot_workload_bars"),
        (FigureKind.ONLINE_TIME, MetricId.ONLINE_S, "plot_workload_bars"),
        (FigureKind.COST_BREAKDOWN, MetricId.ONLINE_S, "plot_cost_breakdown"),
    ],
)
def test_bar_figures_are_drawn_from_the_summary_frame_they_store(
    report: RecipeProfileReport,
    plot_calls: PlotCalls,
    kind: FigureKind,
    metric: MetricId,
    function: str,
) -> None:
    _draw(report, kind, metric)

    (drawn,) = _frames_of(plot_calls.by_function[function])
    recipe = group_recipe(kind)
    assert_frame_equal(
        drawn,
        figure_summaries(report, no_offline_cost(report))[
            recipe.variant(metric, Population.OWN).stem
        ],
    )


@pytest.mark.parametrize(
    ("kind", "metric", "total"),
    [
        (FigureKind.SOLVE_TIME, MetricId.SOLVE_S, WorkloadTotal.SOLVE),
        (FigureKind.ONLINE_TIME, MetricId.ONLINE_S, WorkloadTotal.ONLINE),
    ],
)
def test_the_time_bars_draw_the_total_of_their_metric(
    report: RecipeProfileReport,
    plot_calls: PlotCalls,
    kind: FigureKind,
    metric: MetricId,
    total: WorkloadTotal,
) -> None:
    _draw(report, kind, metric)

    assert plot_calls.by_function["plot_workload_bars"].options["total"] is total


@pytest.mark.parametrize(
    ("kind", "metric", "statistic"),
    [
        (FigureKind.ITERATION_MEANS, MetricId.ITERATIONS, Statistic.MEAN_OVER_SYSTEMS),
        (FigureKind.TIME_MEANS, MetricId.SOLVE_S, Statistic.MEAN_OVER_SYSTEMS),
        (FigureKind.TIME_MEANS, MetricId.ONLINE_S, Statistic.MEAN_OVER_SYSTEMS),
        (FigureKind.TIME_MEANS, MetricId.STANDALONE_S, Statistic.MEAN_OVER_SYSTEMS),
        (FigureKind.SETUP_MEANS, MetricId.SETUP_S, Statistic.MEAN_OVER_MATRICES),
    ],
)
def test_mean_bars_are_drawn_from_the_summary_they_store_with_their_statistic(
    report: RecipeProfileReport,
    plot_calls: PlotCalls,
    kind: FigureKind,
    metric: MetricId,
    statistic: Statistic,
) -> None:
    _draw(report, kind, metric)

    call = plot_calls.by_function["plot_mean_bars"]
    (drawn,) = _frames_of(call)
    assert_frame_equal(
        drawn, figure_summaries(report, no_offline_cost(report))[_stem(kind, metric)]
    )
    assert (call.options["statistic"], call.options["metric"]) == (statistic, metric)


def test_time_per_iteration_is_drawn_from_its_ratio_table_and_the_observations(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    _draw(report, FigureKind.TIME_PER_ITERATION, MetricId.TIME_PER_ITERATION_S)

    drawn, observations = _frames_of(plot_calls.by_function["plot_time_per_iteration"])
    assert_frame_equal(
        drawn,
        figure_summaries(report, no_offline_cost(report))[
            _stem(FigureKind.TIME_PER_ITERATION, MetricId.TIME_PER_ITERATION_S)
        ],
    )
    assert_frame_equal(observations, report.observations)


def _stem(kind: FigureKind, metric: MetricId) -> str:
    return group_recipe(kind).variant(metric, Population.OWN).stem


@pytest.mark.parametrize(
    ("kind", "metric"),
    [
        (FigureKind.ITERATION_DISTRIBUTION, MetricId.ITERATIONS),
        (FigureKind.TIME_DISTRIBUTION, MetricId.SOLVE_S),
        (FigureKind.TIME_DISTRIBUTION, MetricId.ONLINE_S),
    ],
)
def test_a_distribution_is_drawn_from_its_cohort_summary_and_the_observations(
    report: RecipeProfileReport, plot_calls: PlotCalls, kind: FigureKind, metric: MetricId
) -> None:
    _draw(report, kind, metric)

    call = plot_calls.by_function["plot_distribution"]
    drawn, observations = _frames_of(call)
    assert_frame_equal(
        drawn, figure_summaries(report, no_offline_cost(report))[_stem(kind, metric)]
    )
    assert_frame_equal(observations, report.observations)
    assert call.options["metric"] is metric


def test_the_matrix_heatmap_is_drawn_from_the_breakdown_it_stores(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    _draw(report, FigureKind.MATRIX_HEATMAP, MetricId.ITERATIONS)

    call = plot_calls.by_function["plot_matrix_heatmap"]
    (drawn,) = _frames_of(call)
    assert_frame_equal(
        drawn,
        figure_summaries(report, no_offline_cost(report))[
            _stem(FigureKind.MATRIX_HEATMAP, MetricId.ITERATIONS)
        ],
    )
    assert call.options["metric"] is MetricId.ITERATIONS
