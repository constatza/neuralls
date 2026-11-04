"""One cell per method and matrix: the mean of a per-system metric over the matrix's RHS.

A pooled statistic hides on which matrices a method struggles; the breakdown keeps each matrix
apart. A cell's mean runs over the RHS on which the metric is defined (``summary.select``), so a
failed RHS is never a zero or an iteration cap, and ``n_defined`` against ``n_planned`` says how
many of the matrix's planned RHS it rests on. A cell where nothing is defined keeps its counts and
has no mean.
"""

from __future__ import annotations

import polars as pl

from neuralls.domain.solver.reporting.metrics import MetricId, system_metric
from neuralls.domain.solver.reporting.summary import Population, select

MATRIX_BREAKDOWN_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "metric": pl.String,
    "dataset": pl.String,
    "matrix_id": pl.Int64,
    "n_planned": pl.Int64,
    "n_defined": pl.Int64,
    "mean": pl.Float64,
}
"""One row per method and planned matrix, sorted by method, dataset and matrix; ``mean`` is null
when no RHS of the matrix defines the metric for the method."""

_CELL_KEYS = ["method", "dataset", "matrix_id"]


def matrix_breakdown(obs: pl.DataFrame, metric: MetricId) -> pl.DataFrame:
    """Per method and matrix of ``obs``, the mean of ``metric`` over the matrix's defined RHS.

    Each method's cell is taken over its own systems: a common population would empty every cell
    of a matrix on which any method failed, which is exactly what the breakdown is meant to show.

    Raises:
        ValueError: ``metric`` has one value per matrix, so there is nothing to average over RHS.
    """
    spec = system_metric(metric)
    planned = obs.group_by(_CELL_KEYS).agg(n_planned=pl.len())
    defined = (
        select(obs, metric, Population.OWN)
        .group_by(_CELL_KEYS)
        .agg(n_defined=pl.len(), mean=pl.col(spec.column).cast(pl.Float64).mean())
    )
    return (
        planned.join(defined, on=_CELL_KEYS, how="left")
        .with_columns(pl.col("n_defined").fill_null(0), metric=pl.lit(str(metric)))
        .select(list(MATRIX_BREAKDOWN_SCHEMA))
        .cast(pl.Schema(MATRIX_BREAKDOWN_SCHEMA))
        .sort(_CELL_KEYS)
    )
