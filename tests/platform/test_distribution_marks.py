"""The marks of distribution figures and of the matrix heatmap are read from their stored tables.

A method's median tick is the median of its summary row and its count annotation the row's
``n_defined`` and ``n_planned``; the box, the violin density, the beeswarm and the per-matrix
points, which no table stores, come from the observations. A heatmap cell's annotation and its
hatching are read from the breakdown row of its method and matrix. No figure is drawn.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from matplotlib.cbook import boxplot_stats
from polars.testing import assert_frame_equal
from tests.figure_support import no_offline_cost

from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.summary import Population, select, summarize
from neuralls.platform.reporting.figure_summaries import figure_summaries
from neuralls.platform.reporting.seaborn_distributions import (
    CellMean,
    DistributionPanel,
    HeatmapCell,
    MethodDistribution,
    NoneDefined,
    NoQuantiles,
    Quantiles,
    cell_label,
    distribution_panel,
    heatmap_cells,
    matrix_points,
)
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName

AMG = PreconditionerName("amg")
BROKEN = PreconditionerName("broken")
JACOBI = PreconditionerName("jacobi")
DATASET = DatasetId("ds")


def _methods(panel: DistributionPanel) -> dict[str, MethodDistribution]:
    return {method.method: method for method in panel.methods}


def _cell(cells: tuple[HeatmapCell, ...], method: str, matrix: int) -> HeatmapCell:
    (cell,) = (cell for cell in cells if (cell.method, cell.matrix_id) == (method, matrix))
    return cell


@pytest.fixture
def heatmap(profile_report: RecipeProfileReport) -> tuple[HeatmapCell, ...]:
    return heatmap_cells(
        figure_summaries(profile_report, no_offline_cost(profile_report))[
            "matrix_heatmap_iterations"
        ]
    )


def test_a_methods_quantiles_are_those_of_its_summary_row(
    profile_report: RecipeProfileReport,
) -> None:
    summary = figure_summaries(profile_report, no_offline_cost(profile_report))[
        "iteration_distribution"
    ]

    panel = distribution_panel(summary)

    row = summary.filter(pl.col("method") == JACOBI).row(0, named=True)
    assert _methods(panel)[JACOBI] == MethodDistribution(
        JACOBI,
        Quantiles(row["q05"], row["q25"], row["q50"], row["q75"], row["q95"], row["max"]),
        n_defined=9,
        n_planned=9,
    )


def test_a_method_with_nothing_defined_has_no_violin_but_keeps_its_counts(
    profile_report: RecipeProfileReport,
) -> None:
    summary = figure_summaries(profile_report, no_offline_cost(profile_report))[
        "time_distribution_online_amortized"
    ]

    panel = distribution_panel(summary)

    assert _methods(panel)[BROKEN] == MethodDistribution(
        BROKEN, NoQuantiles(), n_defined=0, n_planned=9
    )


def test_the_points_of_a_panel_are_the_per_matrix_means_of_its_systems(
    profile_report: RecipeProfileReport,
) -> None:
    observations = profile_report.observations
    summary = figure_summaries(profile_report, no_offline_cost(profile_report))[
        "iteration_distribution"
    ]

    points = matrix_points(observations, distribution_panel(summary), MetricId.ITERATIONS)

    expected = (
        select(observations, MetricId.ITERATIONS, Population.OWN)
        .group_by("method", "dataset", "matrix_id", maintain_order=True)
        .agg(value=pl.col("iterations").cast(pl.Float64).mean())
        .select("method", "value")
    )
    assert_frame_equal(points, expected)


def test_a_cell_where_some_rhs_converged_holds_their_mean_and_count(
    heatmap: tuple[HeatmapCell, ...],
) -> None:
    assert _cell(heatmap, AMG, 2) == HeatmapCell(
        AMG, DATASET, MatrixId(2), CellMean(8.0), n_defined=1, n_planned=3
    )


def test_a_cell_where_nothing_converged_has_no_mean(heatmap: tuple[HeatmapCell, ...]) -> None:
    assert _cell(heatmap, AMG, 3) == HeatmapCell(
        AMG, DATASET, MatrixId(3), NoneDefined(), n_defined=0, n_planned=2
    )


@pytest.mark.parametrize(
    ("method", "matrix", "label"),
    [
        (JACOBI, 1, "12"),
        (AMG, 1, "3.33"),
        (AMG, 2, "8\n1/3"),
        (AMG, 3, "0/2 solved"),
        (BROKEN, 4, "0/1 solved"),
    ],
)
def test_a_cell_is_labelled_with_its_mean_and_the_rhs_it_rests_on(
    heatmap: tuple[HeatmapCell, ...], method: str, matrix: int, label: str
) -> None:
    assert cell_label(_cell(heatmap, method, matrix), MetricId.ITERATIONS) == label


def test_the_boxs_quantiles_equal_the_stored_tables_for_the_same_sample(
    profile_report: RecipeProfileReport,
) -> None:
    """Seaborn recomputes the box from the systems; the table must say the same numbers."""
    observations = profile_report.observations
    table = summarize(observations, MetricId.ITERATIONS, Population.OWN)
    row = table.filter(pl.col("method") == JACOBI).row(0, named=True)
    sample = select(observations, MetricId.ITERATIONS, Population.OWN).filter(
        pl.col("method") == JACOBI
    )["iterations"]

    (box,) = boxplot_stats(sample.to_numpy(), whis=(5, 95))

    stored = [row["q05"], row["q25"], row["q50"], row["q75"], row["q95"]]
    assert stored == pytest.approx(np.percentile(sample.to_numpy(), [5, 25, 50, 75, 95]))
    assert [box["q1"], box["med"], box["q3"]] == pytest.approx(stored[1:4])
    # matplotlib ends a whisker on the most extreme system inside the percentile range
    inside = sample.filter(sample.is_between(row["q05"], row["q95"]))
    assert (box["whislo"], box["whishi"]) == (inside.min(), inside.max())
