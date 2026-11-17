r"""Offline break-even $N^\star$: the pair count from which a method's offline cost has paid off.

The unit is a (matrix, RHS) **pair** charged its matrix's whole setup, so a pair's online time is
$t^{\mathrm{online}}_{m,i,q} = s_{m,i} + t_{m,i,q}$ (``online_s``). A method $m$ with offline cost
$C_m$ (``offline``: dataset generation plus fit) serving $N$ pairs costs
$C_m + N\,\bar t^{\mathrm{online}}_m$; a compared method $h$ costs $C_h + N\,\bar t^{\mathrm{online}}_h$.
Only $m$'s offline cost is charged: $N^\star_{m,h}$ answers "after how many pairs has $m$'s
offline investment paid for itself against $h$", which for a classical $h$ (no offline stage) is
the whole question, and for a fitted $h$ is asked again with the roles swapped. Setting
$C_m + N\,\bar t_m = N\,\bar t_h$ gives

$$N^\star_{m,h} = \frac{C_m}{\bar t^{\mathrm{online}}_h - \bar t^{\mathrm{online}}_m},$$

defined only when the denominator, the online saving per pair, is positive; otherwise $m$ never
catches up (`Never`). Both means are over the pairs **both** methods converged, so neither side is
flattered by pairs the other failed.

The count is in pairs of serial compute, with no worker count $n_w$: offline generation (one
independent snapshot solve per row) and online pair solves are both embarrassingly parallel, so
dividing only one side by a number of workers would favour that side for a reason unrelated to
the methods; serial compute charges both the same way.

The interval is the two-level bootstrap of ``domain/benchmarking/uncertainty.py`` (matrices, then
each event's timing repeats) of the saving $D = \bar t_h - \bar t_m$, the mean over systems of
each pair's difference. $N^\star = g(D)$ with $g(D) = C_m / D$ for $D > 0$ and $g(D) = +\infty$
(never) otherwise is non-increasing in $D$, so the percentile interval of $N^\star$ over the same
resamples is the image of $D$'s, its ends swapped: $[g(D_{97.5}), g(D_{2.5})]$, up to the
interpolation between neighbouring resamples. A resample in which $h$ is no slower than $m$ never
breaks even and counts as $+\infty$: when more than 2.5% of resamples do, the upper bound is
$+\infty$, which states that the data are consistent with never breaking even, rather than
dropping those resamples (which would flatter $m$) or the interval (which would hide them).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

import polars as pl

from neuralls.domain.benchmarking.cost_curves import BreakEvenState
from neuralls.domain.benchmarking.uncertainty import ClusterStatistic, two_level_interval
from neuralls.domain.solver.models.cost import (
    NO_OFFLINE_COST,
    Known,
    NotApplicable,
    OfflineStages,
    Unavailable,
)
from neuralls.domain.solver.models.statistics import Interval
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.domain.solver.reporting.summary import Population, select
from neuralls.domain.solver.reporting.timed_values import (
    ONLINE_TIMING,
    REFERENCE_SUFFIX,
    TIMING_COLUMNS,
    timed_difference_clusters,
    with_timing_weights,
)
from neuralls.shared.constants import DEFAULT_RANDOM_SEED
from neuralls.shared.types import PreconditionerName

BREAKEVEN_FULL_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "reference_method": pl.String,
    "offline_s": pl.Float64,
    "online_mean_s": pl.Float64,
    "reference_online_mean_s": pl.Float64,
    "break_even": pl.String,
    "breakeven_full": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
    "n_pairs": pl.Int64,
    "n_matrices": pl.Int64,
}
r"""One row per method ``method`` with a known offline cost and every other method
``reference_method`` it is compared against (not the configured reference: every pair of methods
is compared). ``offline_s`` is $C_m$; ``online_mean_s`` and ``reference_online_mean_s`` the mean
``online_s`` of each side over the ``n_pairs`` pairs both converged, on ``n_matrices`` matrices.
``break_even`` holds a ``BreakEvenState``; ``breakeven_full`` is $N^\star$ in pairs and is null
unless it is ``breaks_even``. ``ci_low`` and ``ci_high`` bound $N^\star$ (``+inf`` where a
bootstrap quantile never breaks even) and are null below ``MIN_INTERVAL_MATRICES`` matrices."""

_SYSTEM_KEYS = ["dataset", "system_id"]
_MATRIX_KEYS = ["dataset", "matrix_id"]
_REFERENCE_ONLINE = "reference_online_s"


@dataclass(frozen=True, slots=True)
class BreaksEven:
    """The method costs less than the compared one from ``pairs`` pairs on."""

    pairs: float


@dataclass(frozen=True, slots=True)
class Never:
    """The method is not faster per pair, so its offline cost is never paid off."""


type BreakEvenFull = BreaksEven | Never


def breakeven_count(offline_s: float, saving_s: float) -> BreakEvenFull:
    r"""$N^\star = C_m / D$ for an online saving per pair ``saving_s`` $= D > 0$, else `Never`."""
    if saving_s <= 0.0:
        return Never()
    return BreaksEven(offline_s / saving_s)


def count_bounds(offline_s: float, saving: Interval) -> tuple[float, float]:
    r"""The $N^\star$ interval of a saving interval: ends swapped, ``inf`` where it never breaks even."""
    return _extended_count(offline_s, saving.high), _extended_count(offline_s, saving.low)


def _extended_count(offline_s: float, saving_s: float) -> float:
    match breakeven_count(offline_s, saving_s):
        case BreaksEven(pairs=pairs):
            return pairs
        case Never():
            return math.inf
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True, eq=False)
class FullComparison:
    """Method ``method``, of offline cost ``offline_s``, against ``reference_method``.

    ``pairs`` holds one row per pair both converged: ``dataset``, ``matrix_id``, ``system_id``,
    each side's ``online_s`` and the ``TIMING_COLUMNS`` the two-level bootstrap redraws, the
    compared method's suffixed ``_reference``.
    """

    method: PreconditionerName
    reference_method: PreconditionerName
    offline_s: float
    pairs: pl.DataFrame

    @property
    def n_matrices(self) -> int:
        """The matrices the pairs lie on, the clusters the interval resamples."""
        return self.pairs.unique(subset=_MATRIX_KEYS).height


def full_comparisons(
    obs: pl.DataFrame, offline: Mapping[PreconditionerName, OfflineStages]
) -> tuple[FullComparison, ...]:
    """Every method of ``obs`` with a known offline cost against every other, ordered by name.

    ``offline`` holds the offline stages of the methods of ``obs``. A method whose combined
    offline cost is unavailable (a stage applies but was not recorded), does not apply, or is not
    given has no comparison as ``method``: there is no $C_m$ to pay off. It is still compared
    against as ``reference_method``.
    """
    methods = sorted(PreconditionerName(name) for name in obs["method"].unique())
    comparisons: list[FullComparison] = []
    for method in methods:
        match offline.get(method, NO_OFFLINE_COST).combined():
            case Known(wall_time_seconds=offline_s):
                comparisons.extend(
                    FullComparison(method, other, offline_s, _common_pairs(obs, method, other))
                    for other in methods
                    if other != method
                )
            case NotApplicable() | Unavailable():
                pass
            case unreachable:
                assert_never(unreachable)
    return tuple(comparisons)


def _common_pairs(
    obs: pl.DataFrame, method: PreconditionerName, other: PreconditionerName
) -> pl.DataFrame:
    both = obs.filter(pl.col("method").is_in([method, other]))
    rows = with_timing_weights(select(both, MetricId.ONLINE_S, Population.COMMON), ONLINE_TIMING)
    columns = ["online_s", *TIMING_COLUMNS]
    is_method = pl.col("method") == method
    reference = rows.filter(~is_method).select(
        *_SYSTEM_KEYS,
        *(pl.col(column).alias(f"{column}{REFERENCE_SUFFIX}") for column in TIMING_COLUMNS),
        pl.col("online_s").alias(_REFERENCE_ONLINE),
    )
    return (
        rows.filter(is_method)
        .select(*_MATRIX_KEYS, "system_id", *columns)
        .join(reference, on=_SYSTEM_KEYS, how="inner")
    )


def breakeven_full(
    obs: pl.DataFrame,
    offline: Mapping[PreconditionerName, OfflineStages],
    *,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pl.DataFrame:
    r"""$N^\star$ of every `full_comparisons` entry, with ``BREAKEVEN_FULL_SCHEMA``."""
    rows = [_row(comparison, seed) for comparison in full_comparisons(obs, offline)]
    return pl.DataFrame(rows, schema=BREAKEVEN_FULL_SCHEMA)


def _row(comparison: FullComparison, seed: int) -> dict[str, object]:
    pairs = comparison.pairs
    identity = {
        "method": comparison.method,
        "reference_method": comparison.reference_method,
        "offline_s": comparison.offline_s,
        "n_pairs": pairs.height,
        "n_matrices": comparison.n_matrices,
    }
    if pairs.height == 0:
        return identity | {"break_even": BreakEvenState.NO_COMMON_SYSTEMS.value}
    online_mean_s, reference_mean_s = pairs.select(
        pl.col("online_s").mean(), pl.col(_REFERENCE_ONLINE).mean()
    ).row(0)
    interval = two_level_interval(
        timed_difference_clusters(pairs), ClusterStatistic.MEAN_OVER_SYSTEMS, seed=seed
    )
    low, high = (None, None) if interval is None else count_bounds(comparison.offline_s, interval)
    match breakeven_count(comparison.offline_s, reference_mean_s - online_mean_s):
        case BreaksEven(pairs=count):
            state, breakeven = BreakEvenState.BREAKS_EVEN, count
        case Never():
            state, breakeven = BreakEvenState.NEVER, None
        case unreachable:
            assert_never(unreachable)
    return identity | {
        "online_mean_s": online_mean_s,
        "reference_online_mean_s": reference_mean_s,
        "break_even": state.value,
        "breakeven_full": breakeven,
        "ci_low": low,
        "ci_high": high,
    }
