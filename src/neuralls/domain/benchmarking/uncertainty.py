"""Seeded percentile-bootstrap intervals over whole matrix clusters.

Two bootstraps share one resampling core. The matrix-cluster bootstrap redraws whole matrices
with every sibling value, which is all the uncertainty an iteration count has: it is
deterministic per pair. A time also carries timing noise, so the two-level bootstrap redraws the
matrices and then, inside every drawn matrix, the timing repeats of each event (a setup, a
solve) that its values were reduced from, reducing them again by the event's own
``TimingStatistic`` before the statistic, so the interval is for the estimator the point is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import numpy as np

from neuralls.domain.solver.models.statistics import Interval
from neuralls.shared.constants import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
from neuralls.shared.timing import TimingStatistic
from neuralls.shared.types import MatrixId

type MatrixCluster = MatrixId | tuple[str | None, MatrixId]

MIN_INTERVAL_MATRICES = 5
"""Fewer matrices than this give no interval. A percentile bootstrap undercovers below about ten
clusters, and the fewer the clusters the fewer the distinct resamples (two matrices give only
three), so an interval over a handful of matrices would claim a precision it does not have."""


class ClusterStatistic(StrEnum):
    """What is recomputed on every matrix-cluster resample."""

    MEAN_OVER_SYSTEMS = "mean_over_systems"
    MEAN_OVER_MATRICES = "mean_over_matrices"
    MEDIAN = "median"


class BootstrapKind(StrEnum):
    """What a bootstrap resamples: whole matrices, or matrices and then their timing repeats."""

    MATRIX_CLUSTER = "matrix_cluster_bootstrap"
    TWO_LEVEL = "two_level_bootstrap"

    @property
    def description(self) -> str:
        """The resampled levels in words, as a figure card names the interval."""
        match self:
            case BootstrapKind.MATRIX_CLUSTER:
                return "matrix-cluster bootstrap (matrices)"
            case BootstrapKind.TWO_LEVEL:
                return "two-level bootstrap (matrices, timing repeats)"
            case unreachable:
                assert_never(unreachable)

    @property
    def interval_name(self) -> str:
        """The short name of an interval of this bootstrap, as a figure legend names it."""
        match self:
            case BootstrapKind.MATRIX_CLUSTER:
                return "matrix-cluster interval"
            case BootstrapKind.TWO_LEVEL:
                return "two-level interval"
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class ClusterBootstrap:
    """The bootstrap an interval was computed with, as a figure that draws its interval records it.

    ``level`` and ``resamples`` default to the constants the interval is computed with, so a
    freshly built record states the bootstrap actually used. ``matrices`` is the number of whole
    matrices resampled, the cluster count the interval rests on.
    """

    kind: BootstrapKind
    matrices: int
    level: float = CONFIDENCE_LEVEL
    resamples: int = BOOTSTRAP_RESAMPLES

    def __post_init__(self) -> None:
        """Require enough matrices to resample, a level in (0, 1) and at least one resample."""
        if self.matrices < MIN_INTERVAL_MATRICES:
            raise ValueError(f"a matrix-cluster bootstrap needs {MIN_INTERVAL_MATRICES} matrices")
        if not 0.0 < self.level < 1.0:
            raise ValueError("a bootstrap confidence level lies strictly between 0 and 1")
        if self.resamples < 1:
            raise ValueError("a bootstrap draws at least one resample")


@dataclass(frozen=True, slots=True)
class TooFewMatrices:
    """Fewer than ``MIN_INTERVAL_MATRICES`` matrices gave a value, so no interval is drawn."""

    matrices: int

    def __post_init__(self) -> None:
        """Require a matrix count from 0 up to, not including, ``MIN_INTERVAL_MATRICES``."""
        if not 0 <= self.matrices < MIN_INTERVAL_MATRICES:
            raise ValueError(f"too few matrices means 0 to {MIN_INTERVAL_MATRICES - 1}")


def cluster_bootstrap(matrices: int, kind: BootstrapKind) -> TooFewMatrices | ClusterBootstrap:
    """The ``kind`` bootstrap over ``matrices`` whole matrices, or why there is none."""
    if matrices < MIN_INTERVAL_MATRICES:
        return TooFewMatrices(matrices)
    return ClusterBootstrap(kind=kind, matrices=matrices)


@dataclass(frozen=True, slots=True)
class TimedEvent:
    """One timed event (a setup, a solve): its stored repeats and the statistic reducing them.

    Attributes:
        samples: The stored wall times of the event's timing repeats.
        statistic: The reduction the event's stored time was computed with.
    """

    samples: tuple[float, ...]
    statistic: TimingStatistic

    def __post_init__(self) -> None:
        """Require at least one repeat."""
        if not self.samples:
            raise ValueError("every timed event needs at least one timing repeat")


@dataclass(frozen=True, slots=True, eq=False)
class TimedCluster:
    """One matrix's values as a function of the timing repeats of the events behind them.

    Attributes:
        events: Every event the values read (a setup, a solve).
        values_of: Maps an ``(n, len(events))`` array of event times to the ``(n, rows)``
            values those times give, one row per value of the matrix.
    """

    events: tuple[TimedEvent, ...]
    values_of: Callable[[np.ndarray], np.ndarray]


type _Draw = Callable[[int, np.random.Generator], np.ndarray]
"""Draws ``n`` replicates of one matrix's values, as an ``(n, rows)`` array."""


def matrix_cluster_interval(
    values_by_matrix: Mapping[MatrixCluster, Sequence[float]],
    statistic: ClusterStatistic,
    *,
    seed: int,
) -> Interval | None:
    """Percentile interval of ``statistic`` resampling matrices with all their sibling values.

    ``None`` below ``MIN_INTERVAL_MATRICES`` matrices. Equal values give a degenerate interval.

    Raises:
        ValueError: A matrix has no values.
    """
    if isinstance(
        cluster_bootstrap(len(values_by_matrix), BootstrapKind.MATRIX_CLUSTER), TooFewMatrices
    ):
        return None
    clusters = tuple(np.asarray(values, dtype=np.float64) for values in values_by_matrix.values())
    if any(cluster.size == 0 for cluster in clusters):
        raise ValueError("matrix clusters must contain at least one sample")
    all_values = np.concatenate(clusters)
    if np.all(all_values == all_values[0]):
        value = float(all_values[0])
        return Interval(value, value, CONFIDENCE_LEVEL)
    return _percentile_interval(tuple(_fixed(cluster) for cluster in clusters), statistic, seed)


def two_level_interval(
    clusters: Mapping[MatrixCluster, TimedCluster],
    statistic: ClusterStatistic,
    *,
    seed: int,
) -> Interval | None:
    """Percentile interval of ``statistic`` resampling matrices, then each event's timing repeats.

    Every resample (a) draws matrices with replacement, (b) within each drawn matrix redraws the
    stored repeats of every event with replacement and reduces them by the event's
    ``TimingStatistic``, and (c) recomputes the matrix's values from those times and then
    ``statistic``. An event timed once has
    nothing to redraw, and with no spread in any event's repeats the interval is the
    matrix-cluster one. ``None`` below ``MIN_INTERVAL_MATRICES`` matrices.
    """
    if isinstance(cluster_bootstrap(len(clusters), BootstrapKind.TWO_LEVEL), TooFewMatrices):
        return None
    return _percentile_interval(
        tuple(_timed(cluster) for cluster in clusters.values()), statistic, seed
    )


def _fixed(values: np.ndarray) -> _Draw:
    """A matrix whose values are the same on every draw."""
    return lambda n, rng: np.broadcast_to(values, (n, values.size))


def _timed(cluster: TimedCluster) -> _Draw:
    """A matrix whose values are recomputed from a fresh redraw of its events' repeats."""
    events = cluster.events

    def draw(n: int, rng: np.random.Generator) -> np.ndarray:
        values = cluster.values_of(_event_times(events, n, rng))
        if values.ndim != 2 or values.shape[0] != n or values.shape[1] == 0:
            raise ValueError("a timed cluster must give at least one value per draw")
        return values

    return draw


def _event_times(events: tuple[TimedEvent, ...], n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` reductions of each event's repeats redrawn with replacement, as ``(n, events)``.

    Each redraw is reduced by its event's statistic. Events with the same repeat count and
    statistic are redrawn together; one timed once keeps its time.
    """
    times = np.empty((n, len(events)))
    batches = {(len(event.samples), event.statistic) for event in events}
    for size, statistic in sorted(batches):
        columns = [
            index
            for index, event in enumerate(events)
            if (len(event.samples), event.statistic) == (size, statistic)
        ]
        stacked = np.array([events[index].samples for index in columns], dtype=np.float64)
        if size == 1:
            times[:, columns] = stacked[:, 0]
            continue
        picks = rng.integers(0, size, size=(n, len(columns), size))
        rows = np.arange(len(columns))[:, None]
        times[:, columns] = statistic.reduce_along(stacked[rows, picks], axis=2)
    return times


def _percentile_interval(
    draws: tuple[_Draw, ...], statistic: ClusterStatistic, seed: int
) -> Interval:
    """The percentile interval of ``statistic`` over ``BOOTSTRAP_RESAMPLES`` matrix resamples."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(draws), size=(BOOTSTRAP_RESAMPLES, len(draws)))
    drawn = tuple(draw(int((picks == index).sum()), rng) for index, draw in enumerate(draws))
    estimates = _resampled_statistic(drawn, picks, statistic)
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return Interval(float(low), float(high), CONFIDENCE_LEVEL)


def _gather(per_cluster: Sequence[np.ndarray], picks: np.ndarray) -> np.ndarray:
    """Place each cluster's per-draw values at the resample positions that drew it."""
    gathered = np.empty(picks.shape)
    for index, values in enumerate(per_cluster):
        gathered[picks == index] = values
    return gathered


def _resampled_statistic(
    drawn: tuple[np.ndarray, ...], picks: np.ndarray, statistic: ClusterStatistic
) -> np.ndarray:
    """The statistic of every resample.

    ``picks`` holds the cluster indices drawn per resample and ``drawn[c]`` one row of values
    per time cluster ``c`` was drawn, in the order of its positions in ``picks``.
    """
    match statistic:
        case ClusterStatistic.MEAN_OVER_SYSTEMS:
            sums = _gather([values.sum(axis=1) for values in drawn], picks)
            counts = _gather([np.full(len(values), values.shape[1]) for values in drawn], picks)
            return sums.sum(axis=1) / counts.sum(axis=1)
        case ClusterStatistic.MEAN_OVER_MATRICES:
            return _gather([values.mean(axis=1) for values in drawn], picks).mean(axis=1)
        case ClusterStatistic.MEDIAN:
            slots = _gather([np.arange(len(values)) for values in drawn], picks).astype(np.int64)
            return np.fromiter(
                (
                    np.median(
                        np.concatenate(
                            tuple(drawn[c][slot] for c, slot in zip(selection, row, strict=True))
                        )
                    )
                    for selection, row in zip(picks, slots, strict=True)
                ),
                dtype=np.float64,
                count=len(picks),
            )
        case unreachable:
            assert_never(unreachable)
