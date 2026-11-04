"""The frames and series each group drawer hands to its plot function; no figure is drawn."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
from tests.observation_support import solved_row, synthetic_observations, unconverged_row
from tests.recipe_support import GAUSSIAN, SINE

from neuralls.domain.benchmarking.paired import PairKind
from neuralls.domain.benchmarking.profiles import ProfileSeries
from neuralls.domain.benchmarking.selection import ConfiguredReference, NoReferenceConfigured
from neuralls.domain.solver.models.group_outcomes import (
    GroupAggregated,
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.labels import (
    MatrixSetLabel,
    MatrixSetRhsTypeLabel,
    render_group_id,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.platform.reporting.figure_metadata import FigureKind
from neuralls.platform.reporting.figure_planning import group_recipe
from neuralls.platform.reporting.outcome_plots import GroupStyle
from neuralls.shared.types import DatasetId, PreconditionerName

AMG = PreconditionerName("amg")
JACOBI = PreconditionerName("jacobi")
PATH = Path("figure.png")


@dataclass(slots=True)
class PlotCalls:
    """Arguments the plot functions were called with, by plot function name."""

    by_function: dict[str, dict[str, object]] = field(default_factory=dict)


@pytest.fixture
def plot_calls(monkeypatch: pytest.MonkeyPatch) -> PlotCalls:
    """Replace every plot function with a recorder that returns the path it was given."""
    calls = PlotCalls()

    def recorder(name: str):
        def record(*args: object, save_path: Path, **kwargs: object) -> Path:
            calls.by_function[name] = {"args": args, **kwargs}
            return save_path

        return record

    for name in (
        "plot_ecdf_profile",
        "plot_performance_profile",
        "plot_paired_time_scatter",
        "plot_speed_ratio_cdf",
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


def _draw(report: GroupAggregated, kind: FigureKind, metric: MetricId) -> Path | None:
    return group_recipe(kind).draw(metric, report, MagicMock(spec=GroupStyle), PATH)


def _x(series: ProfileSeries, method: str) -> list[float]:
    return series.frame.filter(pl.col("method") == method)["x"].to_list()


def test_the_iteration_ecdf_is_drawn_from_the_iterations_of_converged_systems(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    assert _draw(report, FigureKind.ECDF_ITERATIONS, MetricId.ITERATIONS) == PATH

    call = plot_calls.by_function["plot_ecdf_profile"]
    (series,) = call["args"]
    assert (_x(series, AMG), _x(series, JACOBI)) == ([3.0, 4.0], [30.0, 40.0])
    assert call["xlabel"] == "Iterations"


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

    (series,) = plot_calls.by_function["plot_ecdf_profile"]["args"]
    assert (_x(series, AMG), _x(series, JACOBI)) == (amg, jacobi)


def test_the_online_performance_profile_is_drawn_from_ratios_to_the_best_method(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    _draw(report, FigureKind.PERFORMANCE_PROFILE, MetricId.ONLINE_S)

    (series,) = plot_calls.by_function["plot_performance_profile"]["args"]
    # s0: amg 1.5 beats jacobi 4.5; s1 only amg; s2 only jacobi
    assert _x(series, AMG) == [1.0, 1.0]
    assert _x(series, JACOBI) == [1.0, 3.0]
    assert series.n_excluded_zero_best == 0


def test_the_paired_scatter_is_drawn_from_the_online_pairs_against_the_reference(
    report: RecipeProfileReport, plot_calls: PlotCalls
) -> None:
    _draw(report, FigureKind.PAIRED_TIME, MetricId.ONLINE_S)

    call = plot_calls.by_function["plot_paired_time_scatter"]
    (paired,) = call["args"]
    assert call["reference_name"] == AMG
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
    ratios, summary = call["args"]
    assert call["reference_name"] == AMG
    assert ratios.select("method", "system_id", "ratio").rows() == [(JACOBI, "s0", 1.5 / 4.5)]
    row = summary.row(0, named=True)
    assert (row["method"], row["n_ratio"], row["n_matrices"]) == (JACOBI, 1, 1)
    assert (row["n_reference_only"], row["n_method_only"]) == (1, 1)


@pytest.mark.parametrize(
    ("kind", "metric"),
    [
        (FigureKind.PAIRED_TIME, MetricId.ONLINE_S),
        (FigureKind.SPEED_RATIO, MetricId.ONLINE_S),
    ],
)
def test_reference_figures_need_a_configured_reference_among_the_methods(
    observations: pl.DataFrame, plot_calls: PlotCalls, kind: FigureKind, metric: MetricId
) -> None:
    unconfigured = _recipe_report(observations, NoReferenceConfigured())
    elsewhere = _recipe_report(observations, ConfiguredReference(PreconditionerName("elsewhere")))

    assert _draw(unconfigured, kind, metric) is None
    assert _draw(elsewhere, kind, metric) is None
    assert plot_calls.by_function == {}


@pytest.mark.parametrize(
    ("kind", "metric"),
    [
        (FigureKind.ECDF_ITERATIONS, MetricId.ITERATIONS),
        (FigureKind.ECDF_TIME, MetricId.ONLINE_S),
        (FigureKind.PERFORMANCE_PROFILE, MetricId.SOLVE_S),
        (FigureKind.PAIRED_TIME, MetricId.ONLINE_S),
        (FigureKind.SPEED_RATIO, MetricId.ONLINE_S),
    ],
)
def test_a_workload_mixture_draws_no_profile_or_pairing(
    observations: pl.DataFrame,
    plot_calls: PlotCalls,
    kind: FigureKind,
    metric: MetricId,
) -> None:
    mixture = WorkloadMixtureReport(
        group_id=render_group_id(MatrixSetLabel(DatasetId("ds"), (GAUSSIAN, SINE))),
        label=MatrixSetLabel(DatasetId("ds"), (GAUSSIAN, SINE)),
        n_matrices=2,
        n_planned_rhs=3,
        observations=observations,
        reference_selection=ConfiguredReference(AMG),
        recipe_membership=(GAUSSIAN, SINE),
    )

    assert _draw(mixture, kind, metric) is None
    assert plot_calls.by_function == {}
