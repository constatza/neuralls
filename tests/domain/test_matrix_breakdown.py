"""The per-matrix breakdown: one cell per method and matrix, failures counted but never averaged."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from neuralls.domain.solver.reporting.matrix_breakdown import (
    MATRIX_BREAKDOWN_SCHEMA,
    matrix_breakdown,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.observations import OBSERVATION_SCHEMA
from tests.observation_support import AMG, BROKEN, DATASET, JACOBI


def _cells(method: str, rows: list[tuple[int, int, int, float | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "method": method,
                "metric": MetricId.ITERATIONS.value,
                "dataset": DATASET,
                "matrix_id": matrix,
                "n_planned": planned,
                "n_defined": defined,
                "mean": mean,
            }
            for matrix, planned, defined, mean in rows
        ],
        schema=MATRIX_BREAKDOWN_SCHEMA,
    )


def test_a_cell_averages_the_converged_rhs_of_its_matrix_and_counts_the_planned_ones(
    unbalanced_observations: pl.DataFrame,
) -> None:
    breakdown = matrix_breakdown(unbalanced_observations, MetricId.ITERATIONS)

    # amg converged on every RHS of matrix 1 and on one of matrix 2; matrix 3 failed its setup
    # and matrix 4 was never reached, so those cells have no mean.
    assert_frame_equal(
        breakdown.filter(pl.col("method") == AMG),
        _cells(AMG, [(1, 3, 3, 10 / 3), (2, 3, 1, 8.0), (3, 2, 0, None), (4, 1, 0, None)]),
    )


def test_every_planned_matrix_of_every_method_has_a_cell(
    unbalanced_observations: pl.DataFrame,
) -> None:
    breakdown = matrix_breakdown(unbalanced_observations, MetricId.ITERATIONS)

    assert breakdown.select("method", "matrix_id").rows() == [
        (method, matrix) for method in sorted((AMG, BROKEN, JACOBI)) for matrix in (1, 2, 3, 4)
    ]


def test_a_method_that_solved_nothing_keeps_its_planned_counts(
    unbalanced_observations: pl.DataFrame,
) -> None:
    breakdown = matrix_breakdown(unbalanced_observations, MetricId.ITERATIONS)

    assert_frame_equal(
        breakdown.filter(pl.col("method") == BROKEN),
        _cells(BROKEN, [(1, 3, 0, None), (2, 3, 0, None), (3, 2, 0, None), (4, 1, 0, None)]),
    )


def test_a_matrix_metric_has_no_per_matrix_mean_over_rhs(
    unbalanced_observations: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        matrix_breakdown(unbalanced_observations, MetricId.SETUP_S)


@pytest.fixture
def no_observations() -> pl.DataFrame:
    return pl.DataFrame(schema=OBSERVATION_SCHEMA)


def test_the_breakdown_of_no_observations_is_empty(no_observations: pl.DataFrame) -> None:
    assert_frame_equal(
        matrix_breakdown(no_observations, MetricId.ITERATIONS),
        pl.DataFrame(schema=MATRIX_BREAKDOWN_SCHEMA),
    )
