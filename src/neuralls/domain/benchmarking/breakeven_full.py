r"""Offline break-even $N^\star$: the pair count at which two methods' full costs are equal.

The unit is a (matrix, RHS) **pair** charged its matrix's whole setup, so a pair's online time is
$t^{\mathrm{online}}_{m,i,q} = s_{m,i} + t_{m,i,q}$ (``online_s``). A method $m$ with offline cost
$C_m$ (``offline``: dataset generation plus fit) serving $N$ pairs costs
$C_m + N\,\bar t^{\mathrm{online}}_m$; a compared method $h$ costs $C_h + N\,\bar t^{\mathrm{online}}_h$,
with $C_h = 0$ when no offline stage applies to $h$ (a classical method) and no comparison at all
when $C_h$ is unknown, which is never read as zero. With the online saving per pair
$D = \bar t^{\mathrm{online}}_h - \bar t^{\mathrm{online}}_m$ and the offline surplus
$\Delta C = C_m - C_h$, the full costs are equal at

$$N^\star_{m,h} = \frac{\Delta C}{D},$$

and which side of $N^\star$ favours $m$ depends on the two signs, a closed set of states:

| $D$ | $\Delta C$ | state | meaning |
|---|---|---|---|
| $> 0$ | $> 0$ | `BreaksEven` | $m$ is cheaper from $N^\star$ pairs on |
| $> 0$ | $\le 0$ | `AlwaysCheaper` | $m$ is no dearer offline and faster per pair |
| $= 0$ | $< 0$ | `AlwaysCheaper` | $m$ is cheaper offline and as fast per pair |
| $< 0$ | $< 0$ | `CheaperUntil` | $m$ is cheaper only below $N^\star$ pairs |
| $\le 0$ | $\ge 0$ | `Never` | $m$ is no cheaper offline and not faster per pair |

Both means are over the pairs **both** methods converged, so neither side is flattered by pairs
the other failed. The formula is antisymmetric, so the comparison of $h$ against $m$ is the mirror
of $m$ against $h$ (`BreaksEven` and `CheaperUntil` swap at the same $N^\star$, `AlwaysCheaper`
and `Never` swap); both orientations of two methods with known offline costs are kept, so each
method's rows read from its own side. A method with no offline stage is compared against
($C_h = 0$) but has no row of its own, since its rows would only mirror those; a method whose
offline cost is unknown is compared in neither role.

The count is in pairs of serial compute, with no worker count $n_w$: offline generation (one
independent snapshot solve per row) and online pair solves are both embarrassingly parallel, so
dividing only one side by a number of workers would favour that side for a reason unrelated to
the methods; serial compute charges both the same way.

The interval is the two-level bootstrap of ``domain/benchmarking/uncertainty.py`` (matrices, then
each event's timing repeats) of $D$, the mean over systems of each pair's difference; $C_m$ and
$C_h$ are single measurements and stay fixed. $\Delta C$ fixes which count is bounded, and that
count is a monotone function of $D$, so the percentile interval of the count over the same
resamples is the image of $D$'s, up to the interpolation between neighbouring resamples:

- $\Delta C \ge 0$: the count from which $m$ is cheaper, $F(D) = \Delta C / D$ for $D > 0$ and
  $+\infty$ (`Never`) otherwise. $F$ is non-increasing, so the bounds are
  $[F(D_{97.5}), F(D_{2.5})]$; an upper bound of $+\infty$ says more than 2.5% of resamples never
  break even. At $\Delta C = 0$, $F = 0$ on $D > 0$: cheaper from the first pair.
- $\Delta C < 0$: the count up to which $m$ is cheaper, $U(D) = \Delta C / D$ for $D < 0$ and
  $+\infty$ (`AlwaysCheaper`) otherwise. $U$ is non-decreasing, so the bounds are
  $[U(D_{2.5}), U(D_{97.5})]$; an upper bound of $+\infty$ says more than 2.5% of resamples are
  always cheaper.

A $D$ interval straddling $0$ straddles the state boundary, and the infinite bound states it,
rather than dropping the resamples across it (which would flatter one side) or the interval
(which would hide them).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import polars as pl

from neuralls.domain.benchmarking.uncertainty import ClusterStatistic, two_level_interval
from neuralls.domain.solver.models.cost import (
    Known,
    NotApplicable,
    OfflineStages,
    StageCost,
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


class FullBreakEvenState(StrEnum):
    """The stored state of a ``breakeven_full`` row: one per `BreakEvenFull` variant, plus one.

    ``NO_COMMON_SYSTEMS``: the two methods converged on no pair in common, so there are no means.
    The per-matrix $n^*$ table has its own ``BreakEvenState``, without the two states that need a
    cheaper offline side.
    """

    BREAKS_EVEN = "breaks_even"
    ALWAYS_CHEAPER = "always_cheaper"
    CHEAPER_UNTIL = "cheaper_until"
    NEVER = "never"
    NO_COMMON_SYSTEMS = "no_common_systems"


BREAKEVEN_FULL_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "method": pl.String,
    "reference_method": pl.String,
    "offline_s": pl.Float64,
    "reference_offline_s": pl.Float64,
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
``reference_method`` whose offline cost is known or does not apply (not the configured reference:
every pair of methods is compared). ``offline_s`` is $C_m$ and ``reference_offline_s`` $C_h$
(0.0 where no offline stage applies); ``online_mean_s`` and ``reference_online_mean_s`` the mean
``online_s`` of each side over the ``n_pairs`` pairs both converged, on ``n_matrices`` matrices.
``break_even`` holds a ``FullBreakEvenState``; ``breakeven_full`` is $N^\star$ in pairs, null
unless it is ``breaks_even`` (cheaper from it on) or ``cheaper_until`` (cheaper below it).
``ci_low`` and ``ci_high`` bound the count from which ``method`` is cheaper when ``offline_s`` is
at least ``reference_offline_s`` (``+inf``: never), else the count up to which it is cheaper
(``+inf``: always); both are null below ``MIN_INTERVAL_MATRICES`` matrices."""

_SYSTEM_KEYS = ["dataset", "system_id"]
_MATRIX_KEYS = ["dataset", "matrix_id"]
_REFERENCE_ONLINE = "reference_online_s"
_MISSING_OFFLINE = Unavailable("no offline cost given")


@dataclass(frozen=True, slots=True)
class BreaksEven:
    """The method is dearer offline but faster per pair: cheaper from ``pairs`` pairs on."""

    pairs: float


@dataclass(frozen=True, slots=True)
class AlwaysCheaper:
    """The method is no dearer offline and no slower per pair, one of them strictly."""


@dataclass(frozen=True, slots=True)
class CheaperUntil:
    """The method is cheaper offline but slower per pair: cheaper only below ``pairs`` pairs."""

    pairs: float


@dataclass(frozen=True, slots=True)
class Never:
    """The method is no cheaper offline and not faster per pair, so it is never cheaper."""


type BreakEvenFull = BreaksEven | AlwaysCheaper | CheaperUntil | Never


def breakeven_count(offline_surplus_s: float, saving_s: float) -> BreakEvenFull:
    r"""The state of $\Delta C$ ``offline_surplus_s`` and $D$ ``saving_s``, per the module table."""
    if saving_s > 0.0:
        if offline_surplus_s > 0.0:
            return BreaksEven(offline_surplus_s / saving_s)
        return AlwaysCheaper()
    if offline_surplus_s < 0.0:
        if saving_s < 0.0:
            return CheaperUntil(offline_surplus_s / saving_s)
        return AlwaysCheaper()
    return Never()


def count_bounds(offline_surplus_s: float, saving: Interval) -> tuple[float, float]:
    r"""The count interval of a saving interval: $F$'s for $\Delta C \ge 0$, else $U$'s.

    $F$ (cheaper from; ``inf``: never) is non-increasing in $D$, so its ends swap; $U$ (cheaper
    until; ``inf``: always) is non-decreasing, so they do not.
    """
    if offline_surplus_s >= 0.0:
        return (
            _cheaper_from(offline_surplus_s, saving.high),
            _cheaper_from(offline_surplus_s, saving.low),
        )
    return (
        _cheaper_until(offline_surplus_s, saving.low),
        _cheaper_until(offline_surplus_s, saving.high),
    )


def _cheaper_from(offline_surplus_s: float, saving_s: float) -> float:
    return offline_surplus_s / saving_s if saving_s > 0.0 else math.inf


def _cheaper_until(offline_surplus_s: float, saving_s: float) -> float:
    return offline_surplus_s / saving_s if saving_s < 0.0 else math.inf


@dataclass(frozen=True, slots=True, eq=False)
class FullComparison:
    """Method ``method``, of offline cost ``offline_s``, against ``reference_method``.

    ``reference_offline_s`` is the compared method's offline cost, 0.0 where none applies.
    ``pairs`` holds one row per pair both converged: ``dataset``, ``matrix_id``, ``system_id``,
    each side's ``online_s`` and the ``TIMING_COLUMNS`` the two-level bootstrap redraws, the
    compared method's suffixed ``_reference``.
    """

    method: PreconditionerName
    reference_method: PreconditionerName
    offline_s: float
    reference_offline_s: float
    pairs: pl.DataFrame

    @property
    def offline_surplus_s(self) -> float:
        r"""$\Delta C = C_m - C_h$."""
        return self.offline_s - self.reference_offline_s

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
    given has no comparison as ``method``. It is compared against as ``reference_method`` when its
    cost is known or does not apply ($C_h = 0$), never when it is unavailable or not given, which
    would read an unknown cost as zero.
    """
    methods = sorted(PreconditionerName(name) for name in obs["method"].unique())
    comparisons: list[FullComparison] = []
    for method in methods:
        match _combined(offline, method):
            case Known(wall_time_seconds=offline_s):
                comparisons.extend(
                    FullComparison(
                        method, other, offline_s, other_s, _common_pairs(obs, method, other)
                    )
                    for other, other_s in _compared(offline, methods, method)
                )
            case NotApplicable() | Unavailable():
                pass
            case unreachable:
                assert_never(unreachable)
    return tuple(comparisons)


def _combined(
    offline: Mapping[PreconditionerName, OfflineStages], method: PreconditionerName
) -> StageCost:
    stages = offline.get(method)
    return _MISSING_OFFLINE if stages is None else stages.combined()


def _compared(
    offline: Mapping[PreconditionerName, OfflineStages],
    methods: Sequence[PreconditionerName],
    method: PreconditionerName,
) -> Iterator[tuple[PreconditionerName, float]]:
    """Every method but ``method`` whose offline cost is known or does not apply, with $C_h$."""
    for other in methods:
        if other == method:
            continue
        match _combined(offline, other):
            case Known(wall_time_seconds=other_s):
                yield other, other_s
            case NotApplicable():
                yield other, 0.0
            case Unavailable():
                pass
            case unreachable:
                assert_never(unreachable)


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
        "reference_offline_s": comparison.reference_offline_s,
        "n_pairs": pairs.height,
        "n_matrices": comparison.n_matrices,
    }
    if pairs.height == 0:
        return identity | {"break_even": FullBreakEvenState.NO_COMMON_SYSTEMS.value}
    online_mean_s, reference_mean_s = pairs.select(
        pl.col("online_s").mean(), pl.col(_REFERENCE_ONLINE).mean()
    ).row(0)
    surplus_s = comparison.offline_surplus_s
    interval = two_level_interval(
        timed_difference_clusters(pairs), ClusterStatistic.MEAN_OVER_SYSTEMS, seed=seed
    )
    low, high = (None, None) if interval is None else count_bounds(surplus_s, interval)
    match breakeven_count(surplus_s, reference_mean_s - online_mean_s):
        case BreaksEven(pairs=count):
            state, breakeven = FullBreakEvenState.BREAKS_EVEN, count
        case CheaperUntil(pairs=count):
            state, breakeven = FullBreakEvenState.CHEAPER_UNTIL, count
        case AlwaysCheaper():
            state, breakeven = FullBreakEvenState.ALWAYS_CHEAPER, None
        case Never():
            state, breakeven = FullBreakEvenState.NEVER, None
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
