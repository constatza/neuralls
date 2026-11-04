"""Populations, the failure rule, and per-method summaries of the observation view.

``select`` is the only place the failure rule is applied: a metric exists for a system only on the
outcomes in its ``MetricSpec.defined_on``, so every statistic here reads the selected rows and a
system that failed is never a zero, an iteration cap or a missing term of a mean.

Vocabulary: a *timing repeat* is the same solve timed again (already reduced inside ``solve_s``),
a *bootstrap resample* is an in-memory redraw of whole matrices, a *RHS* is one system of a matrix.
``mean_over_systems`` pools every selected system; ``mean_over_matrices`` averages the per-matrix
means so every matrix weighs the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import assert_never

import polars as pl

from neuralls.domain.benchmarking.uncertainty import (
    ClusterStatistic,
    MatrixCluster,
    matrix_cluster_interval,
)
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId, MetricSpec, ObservationUnit
from neuralls.domain.solver.reporting.observations import Outcome
from neuralls.shared.constants import DEFAULT_RANDOM_SEED, RESIDUAL_MISMATCH_FACTOR
from neuralls.shared.types import MatrixId

SUMMARY_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "metric": pl.String,
    "population": pl.String,
    "n_planned": pl.Int64,
    "n_defined": pl.Int64,
    "n_matrices": pl.Int64,
    "n_unconverged": pl.Int64,
    "n_solver_error": pl.Int64,
    "n_setup_failed": pl.Int64,
    "n_not_attempted": pl.Int64,
    "mean_over_systems": pl.Float64,
    "mean_over_matrices": pl.Float64,
    "mean_over_systems_ci_low": pl.Float64,
    "mean_over_systems_ci_high": pl.Float64,
    "mean_over_matrices_ci_low": pl.Float64,
    "mean_over_matrices_ci_high": pl.Float64,
    "q05": pl.Float64,
    "q25": pl.Float64,
    "q50": pl.Float64,
    "q75": pl.Float64,
    "q95": pl.Float64,
    "min": pl.Float64,
    "max": pl.Float64,
}
"""One row per method. ``n_planned`` and ``n_defined`` count the metric's observation units (systems
or matrices); the per-outcome counts always count systems, whichever the metric's unit."""

RESIDUAL_CONSISTENCY_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "n_checked": pl.Int64,
    "n_above_threshold": pl.Int64,
    "worst_ratio": pl.Float64,
}

TIMING_NOISE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "median_timing_spread": pl.Float64,
    "n_systems": pl.Int64,
}

_QUANTILE_LEVELS = {"q05": 0.05, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q95": 0.95}
_MATRIX_KEYS = ["dataset", "matrix_id"]
_NO_STATISTICS: dict[str, float | None] = dict.fromkeys(
    (
        "mean_over_systems",
        "mean_over_matrices",
        "mean_over_systems_ci_low",
        "mean_over_systems_ci_high",
        "mean_over_matrices_ci_low",
        "mean_over_matrices_ci_high",
        *_QUANTILE_LEVELS,
        "min",
        "max",
    )
)


class Population(StrEnum):
    """Which systems a method's statistic is taken over."""

    OWN = "own"
    """Every system the method has a value for."""
    COMMON = "common"
    """Only systems (or matrices) every method in the view has a value for."""


def _unit_keys(unit: ObservationUnit) -> list[str]:
    match unit:
        case ObservationUnit.SYSTEM:
            return ["dataset", "system_id"]
        case ObservationUnit.MATRIX:
            return _MATRIX_KEYS
        case unreachable:
            assert_never(unreachable)


def select(obs: pl.DataFrame, metric: MetricId, population: Population) -> pl.DataFrame:
    """The observation rows ``metric`` is defined on, one per observation unit and method.

    A row is kept when its outcome is in the metric's ``defined_on`` and its value exists
    (time per iteration has none at zero iterations). A matrix metric keeps one row per matrix.
    ``Population.COMMON`` keeps only units defined for every method present in ``obs``.
    """
    spec = METRICS[metric]
    keys = _unit_keys(spec.observation_unit)
    defined = obs.filter(
        pl.col("outcome").is_in([outcome.value for outcome in spec.defined_on]),
        pl.col(spec.column).is_not_null(),
    )
    if spec.observation_unit is ObservationUnit.MATRIX:
        defined = defined.unique(subset=["method", *keys], keep="first", maintain_order=True)
    match population:
        case Population.OWN:
            return defined
        case Population.COMMON:
            shared = (
                defined.group_by(keys)
                .agg(n_methods=pl.col("method").n_unique())
                .filter(pl.col("n_methods") == obs["method"].n_unique())
                .select(keys)
            )
            return defined.join(shared, on=keys, how="semi", maintain_order="left")
        case unreachable:
            assert_never(unreachable)


def summarize(
    obs: pl.DataFrame,
    metric: MetricId,
    population: Population,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    """One summary row per method of ``metric`` over ``population``, with ``SUMMARY_SCHEMA``.

    A method with no defined unit still has a row: ``n_defined`` is 0 and every statistic is null,
    never 0. Intervals are 95% matrix-cluster bootstrap intervals, null below two matrices.
    """
    spec = METRICS[metric]
    selected = select(obs, metric, population)
    rows = [
        _summary_row(
            spec,
            population,
            method,
            obs.filter(pl.col("method") == method),
            selected.filter(pl.col("method") == method),
            seed,
        )
        for method in sorted(obs["method"].unique().to_list())
    ]
    return pl.DataFrame(rows, schema=SUMMARY_SCHEMA)


def _summary_row(
    spec: MetricSpec,
    population: Population,
    method: str,
    planned: pl.DataFrame,
    selected: pl.DataFrame,
    seed: int,
) -> dict[str, object]:
    n_planned = (
        planned.height
        if spec.observation_unit is ObservationUnit.SYSTEM
        else planned.unique(subset=_MATRIX_KEYS).height
    )
    outcomes = planned["outcome"]
    return {
        "method": method,
        "metric": str(spec.id),
        "population": str(population),
        "n_planned": n_planned,
        "n_defined": selected.height,
        "n_matrices": selected.unique(subset=_MATRIX_KEYS).height,
        "n_unconverged": int((outcomes == Outcome.UNCONVERGED).sum()),
        "n_solver_error": int((outcomes == Outcome.SOLVER_ERROR).sum()),
        "n_setup_failed": int((outcomes == Outcome.SETUP_FAILED).sum()),
        "n_not_attempted": int((outcomes == Outcome.NOT_ATTEMPTED).sum()),
        **_statistics(spec, selected, seed),
    }


def _statistics(spec: MetricSpec, selected: pl.DataFrame, seed: int) -> dict[str, float | None]:
    if selected.height == 0:
        return dict(_NO_STATISTICS)
    value = pl.col(spec.column).cast(pl.Float64)
    by_matrix = selected.group_by(_MATRIX_KEYS, maintain_order=True).agg(value)
    clusters: dict[MatrixCluster, Sequence[float]] = {
        (dataset, MatrixId(matrix_id)): cluster
        for dataset, matrix_id, cluster in by_matrix.iter_rows()
    }
    systems_ci = matrix_cluster_interval(clusters, ClusterStatistic.MEAN_OVER_SYSTEMS, seed=seed)
    matrices_ci = matrix_cluster_interval(clusters, ClusterStatistic.MEAN_OVER_MATRICES, seed=seed)
    pooled = selected.select(
        mean_over_systems=value.mean(),
        **{
            column: value.quantile(level, interpolation="linear")
            for column, level in _QUANTILE_LEVELS.items()
        },
        min=value.min(),
        max=value.max(),
    ).row(0, named=True)
    per_matrix_mean = by_matrix.select(mean_over_matrices=pl.col(spec.column).list.mean().mean())
    return {
        **pooled,
        **per_matrix_mean.row(0, named=True),
        "mean_over_systems_ci_low": systems_ci.low if systems_ci else None,
        "mean_over_systems_ci_high": systems_ci.high if systems_ci else None,
        "mean_over_matrices_ci_low": matrices_ci.low if matrices_ci else None,
        "mean_over_matrices_ci_high": matrices_ci.high if matrices_ci else None,
    }


def residual_consistency(obs: pl.DataFrame) -> pl.DataFrame:
    """Per method, how far converged systems' true residual sits above the stopping level.

    The solver stops on its recursive residual; the true residual is recomputed independently.
    ``n_above_threshold`` counts systems whose true residual exceeds ``RESIDUAL_MISMATCH_FACTOR``
    times the stopping level and ``worst_ratio`` is the largest true-to-stopping ratio.
    """
    checked = obs.filter(
        pl.col("outcome") == Outcome.CONVERGED.value,
        pl.col("true_residual_rel").is_not_null(),
        pl.col("stopping_threshold_rel").is_not_null(),
    )
    ratio = pl.col("true_residual_rel") / pl.col("stopping_threshold_rel")
    per_method = checked.group_by("method").agg(
        n_checked=pl.len(),
        n_above_threshold=(ratio > RESIDUAL_MISMATCH_FACTOR).sum(),
        worst_ratio=ratio.max(),
    )
    return (
        _per_method(obs, per_method, ["n_checked", "n_above_threshold"])
        .select(list(RESIDUAL_CONSISTENCY_SCHEMA))
        .cast(pl.Schema(RESIDUAL_CONSISTENCY_SCHEMA))
    )


def timing_noise(obs: pl.DataFrame) -> pl.DataFrame:
    """Per method, the median relative spread of timing repeats over systems with a solve row."""
    per_method = (
        obs.filter(pl.col("solve_s").is_not_null())
        .group_by("method")
        .agg(
            median_timing_spread=pl.col("timing_spread").median(),
            n_systems=pl.len(),
        )
    )
    return (
        _per_method(obs, per_method, ["n_systems"])
        .select(list(TIMING_NOISE_SCHEMA))
        .cast(pl.Schema(TIMING_NOISE_SCHEMA))
    )


def _per_method(obs: pl.DataFrame, per_method: pl.DataFrame, counts: list[str]) -> pl.DataFrame:
    """Give every method of ``obs`` a row, with zero counts where it had nothing to aggregate."""
    return (
        obs.select("method")
        .unique()
        .sort("method")
        .join(per_method, on="method", how="left")
        .with_columns(pl.col(counts).fill_null(0))
    )
