"""Failure-aware paired ratios with matrix-clustered uncertainty."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import assert_never

import polars as pl

from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    NoReferenceConfigured,
    ReferenceSelection,
    ReferenceUnavailable,
)
from neuralls.domain.benchmarking.uncertainty import (
    ClusterStatistic,
    MatrixCluster,
    matrix_cluster_interval,
)
from neuralls.domain.solver.reporting.metrics import METRICS, MetricId, system_metric
from neuralls.domain.solver.reporting.summary import Population, select
from neuralls.shared.constants import DEFAULT_RANDOM_SEED
from neuralls.shared.types import MatrixId, PreconditionerName

REFERENCE_METRIC = MetricId.ONLINE_S
"""The metric methods are compared to the configured reference on: amortized online time."""


class PairKind(StrEnum):
    """Which side of a method/reference pair has a value for the metric on a system."""

    BOTH = "both"
    METHOD_ONLY = "method_only"
    REFERENCE_ONLY = "reference_only"


PAIRED_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "system_id": pl.String,
    "dataset": pl.String,
    "matrix_id": pl.Int64,
    "pair_kind": pl.String,
    "reference_value": pl.Float64,
    "method_value": pl.Float64,
}
"""One row per non-reference method and system at least one side has a value for; ``pair_kind``
holds ``PairKind`` values and the value of a side without one is null."""

SPEED_RATIO_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "n_systems_both_solved": pl.Int64,
    "n_ratio": pl.Int64,
    "n_matrices": pl.Int64,
    "n_reference_only": pl.Int64,
    "n_method_only": pl.Int64,
    "median_ratio": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
}

BREAK_EVEN_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "setup_s_mean": pl.Float64,
    "reference_setup_s_mean": pl.Float64,
    "solve_s_mean": pl.Float64,
    "reference_solve_s_mean": pl.Float64,
    "break_even_rhs": pl.Float64,
}

_SYSTEM_KEYS = ["dataset", "system_id"]
_MATRIX_KEYS = ["dataset", "matrix_id"]
_PAIR_ORDER = ["method", "dataset", "matrix_id", "system_id"]
_NO_RATIO_STATISTICS: dict[str, float | None] = {
    "median_ratio": None,
    "ci_low": None,
    "ci_high": None,
}


def reference_among_methods(
    obs: pl.DataFrame, selection: ReferenceSelection
) -> PreconditionerName | None:
    """The configured reference's name when it is among the methods of ``obs``, else ``None``.

    An unconfigured or unavailable reference, and a configured one with no systems in this
    group, give no comparison. The selection is configuration, never inferred from outcomes.
    """
    match selection:
        case ConfiguredReference(name=name) if name in set(obs["method"]):
            return name
        case ConfiguredReference() | NoReferenceConfigured() | ReferenceUnavailable():
            return None
        case unreachable:
            assert_never(unreachable)


def _require_reference(obs: pl.DataFrame, reference: PreconditionerName) -> None:
    if reference not in set(obs["method"]):
        raise ValueError(f"reference {reference!r} is not among the methods of the observations")


def _compared_methods(obs: pl.DataFrame, reference: PreconditionerName) -> list[str]:
    return sorted(set(obs["method"]) - {reference})


def paired_systems(
    obs: pl.DataFrame, reference: PreconditionerName, metric: MetricId
) -> pl.DataFrame:
    """Each non-reference method's systems against the reference, with ``PAIRED_SCHEMA``.

    Only systems planned for both the method and the reference are paired. A pair is kept when at
    least one side has a value for ``metric`` (as ``select`` defines it) and is ``BOTH``,
    ``METHOD_ONLY`` or ``REFERENCE_ONLY`` accordingly; a system neither side has a value for is
    left out. No value is imputed for the side without one.

    Raises:
        ValueError: The reference is not among the methods, or the metric has one value per
            matrix.
    """
    spec = system_metric(metric)
    _require_reference(obs, reference)
    is_reference = pl.col("method") == reference
    values = select(obs, metric, Population.OWN).select(
        "method", *_SYSTEM_KEYS, value=pl.col(spec.column).cast(pl.Float64)
    )
    reference_values = values.filter(is_reference).select(
        *_SYSTEM_KEYS, reference_value=pl.col("value")
    )
    method_values = values.filter(~is_reference).rename({"value": "method_value"})
    reference_planned = obs.filter(is_reference).select(_SYSTEM_KEYS)
    has_reference = pl.col("reference_value").is_not_null()
    has_method = pl.col("method_value").is_not_null()
    return (
        obs.filter(~is_reference)
        .select("method", *_MATRIX_KEYS, "system_id")
        .join(reference_planned, on=_SYSTEM_KEYS, how="inner")
        .join(method_values, on=["method", *_SYSTEM_KEYS], how="left")
        .join(reference_values, on=_SYSTEM_KEYS, how="left")
        .filter(has_reference | has_method)
        .with_columns(
            pair_kind=pl.when(has_reference & has_method)
            .then(pl.lit(PairKind.BOTH.value))
            .when(has_method)
            .then(pl.lit(PairKind.METHOD_ONLY.value))
            .otherwise(pl.lit(PairKind.REFERENCE_ONLY.value))
        )
        .select(list(PAIRED_SCHEMA))
        .cast(pl.Schema(PAIRED_SCHEMA))
        .sort(_PAIR_ORDER)
    )


def speed_ratio_summary(
    obs: pl.DataFrame,
    reference: PreconditionerName,
    metric: MetricId,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    r"""Per non-reference method, the reference-over-method ratio of ``metric`` with its interval.

    The ratio $\rho_s = x_{\mathrm{ref},s} / x_{p,s}$ is taken over the systems both sides have a
    value for and both values are positive, so $\rho_s > 1$ favours the method. A zero value
    makes the ratio undefined: that system still counts in ``n_systems_both_solved`` but not in
    ``n_ratio``. ``n_matrices`` is the number of distinct ``(dataset, matrix_id)`` among the ratio
    rows, the cluster count the interval rests on: sibling RHS share a matrix, so it is not
    ``n_ratio``. ``median_ratio`` and the 95% bootstrap interval (``ci_low``, ``ci_high``) resample
    whole matrices; the statistics are null without a ratio and the interval is null below two
    matrices. ``n_reference_only`` and ``n_method_only`` count the one-sided systems.

    Raises:
        ValueError: The reference is not among the methods, or the metric has one value per
            matrix.
    """
    paired = paired_systems(obs, reference, metric)
    rows = [
        _speed_ratio_row(method, paired.filter(pl.col("method") == method), seed)
        for method in _compared_methods(obs, reference)
    ]
    return pl.DataFrame(rows, schema=SPEED_RATIO_SCHEMA)


def speed_ratios(paired: pl.DataFrame) -> pl.DataFrame:
    r"""The per-system ratios $\rho_s = x_{\mathrm{ref},s} / x_{p,s}$ of a ``paired_systems`` frame.

    Only ``BOTH`` rows whose two values are positive have a ratio; the result keeps ``method``,
    ``dataset``, ``matrix_id`` and ``system_id`` and adds ``ratio``.
    """
    return (
        paired.filter(
            pl.col("pair_kind") == PairKind.BOTH.value,
            pl.col("reference_value") > 0.0,
            pl.col("method_value") > 0.0,
        )
        .with_columns(ratio=pl.col("reference_value") / pl.col("method_value"))
        .select("method", *_MATRIX_KEYS, "system_id", "ratio")
    )


def _speed_ratio_row(method: str, pairs: pl.DataFrame, seed: int) -> dict[str, object]:
    kind = pl.col("pair_kind")
    ratios = speed_ratios(pairs)
    return {
        "method": method,
        "n_systems_both_solved": pairs.filter(kind == PairKind.BOTH.value).height,
        "n_ratio": ratios.height,
        "n_matrices": ratios.unique(subset=_MATRIX_KEYS).height,
        "n_reference_only": pairs.filter(kind == PairKind.REFERENCE_ONLY.value).height,
        "n_method_only": pairs.filter(kind == PairKind.METHOD_ONLY.value).height,
        **_ratio_statistics(ratios, seed),
    }


def _ratio_statistics(ratios: pl.DataFrame, seed: int) -> dict[str, float | None]:
    if ratios.height == 0:
        return dict(_NO_RATIO_STATISTICS)
    by_matrix = ratios.group_by(_MATRIX_KEYS, maintain_order=True).agg(pl.col("ratio"))
    clusters: dict[MatrixCluster, Sequence[float]] = {
        (dataset, MatrixId(matrix_id)): cluster
        for dataset, matrix_id, cluster in by_matrix.iter_rows()
    }
    interval = matrix_cluster_interval(clusters, ClusterStatistic.MEDIAN, seed=seed)
    return {
        "median_ratio": ratios.select(pl.col("ratio").median()).item(),
        "ci_low": interval.low if interval else None,
        "ci_high": interval.high if interval else None,
    }


def break_even(obs: pl.DataFrame, reference: PreconditionerName) -> pl.DataFrame:
    r"""Per non-reference method, the RHS count after which it costs less than the reference.

    With the mean per-matrix setup $\bar{t}^{\mathrm{setup}}$ and the mean per-system solve time
    $\bar{t}^{\mathrm{solve},\mathrm{sys}}$ of each side, a method $p$ costs
    $C_p(n) = \bar{t}^{\mathrm{setup}}_p + n\,\bar{t}^{\mathrm{solve},\mathrm{sys}}_p$ for $n$
    right-hand sides of one matrix, and

    $$n^{*} = \frac{\bar{t}^{\mathrm{setup}}_p - \bar{t}^{\mathrm{setup}}_{\mathrm{ref}}}
    {\bar{t}^{\mathrm{solve},\mathrm{sys}}_{\mathrm{ref}} - \bar{t}^{\mathrm{solve},\mathrm{sys}}_p}$$

    is defined only when the denominator is positive; otherwise the method is not faster per RHS,
    never breaks even, and ``break_even_rhs`` is null. When the denominator is positive and the
    method's setup is no dearer than the reference's, it never costs more and ``break_even_rhs``
    is 0.0.

    The means are taken over the population common to the pair alone (not to every method):
    ``SETUP_S`` over the matrices both set up, ``SOLVE_S`` over the systems both solved, so
    neither side's mean is flattered by systems the other failed. A pair with no common unit has
    null means and a null ``n*``.

    Raises:
        ValueError: The reference is not among the methods.
    """
    _require_reference(obs, reference)
    rows = [_common_means(obs, reference, method) for method in _compared_methods(obs, reference)]
    saving_per_rhs = pl.col("reference_solve_s_mean") - pl.col("solve_s_mean")
    extra_setup = pl.col("setup_s_mean") - pl.col("reference_setup_s_mean")
    return (
        pl.DataFrame(rows, schema=BREAK_EVEN_SCHEMA)
        .with_columns(
            break_even_rhs=pl.when(saving_per_rhs > 0.0).then(
                pl.when(extra_setup > 0.0).then(extra_setup / saving_per_rhs).otherwise(0.0)
            )
        )
        .cast(pl.Schema(BREAK_EVEN_SCHEMA))
    )


def _common_means(
    obs: pl.DataFrame, reference: PreconditionerName, method: str
) -> dict[str, object]:
    """The setup and solve means of the pair on the units both have a value for."""
    pair = obs.filter(pl.col("method").is_in([reference, method]))
    setup = _means_by_method(pair, MetricId.SETUP_S)
    solve = _means_by_method(pair, MetricId.SOLVE_S)
    return {
        "method": method,
        "setup_s_mean": setup.get(method),
        "reference_setup_s_mean": setup.get(reference),
        "solve_s_mean": solve.get(method),
        "reference_solve_s_mean": solve.get(reference),
        "break_even_rhs": None,
    }


def _means_by_method(pair: pl.DataFrame, metric: MetricId) -> dict[str, float]:
    column = METRICS[metric].column
    common = select(pair, metric, Population.COMMON)
    means = common.group_by("method").agg(mean=pl.col(column).mean())
    return dict(zip(means["method"], means["mean"], strict=True))
