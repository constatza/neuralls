"""Seeded percentile-bootstrap intervals over whole matrix clusters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, assert_never

import numpy as np

from neuralls.domain.solver.models.statistics import Interval
from neuralls.shared.constants import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
from neuralls.shared.types import MatrixId

type MatrixCluster = MatrixId | tuple[str | None, MatrixId]

MIN_INTERVAL_MATRICES = 2
"""Fewer matrices than this give no interval: resampling one matrix only repeats it."""


class ClusterStatistic(StrEnum):
    """What is recomputed on every matrix-cluster resample."""

    MEAN_OVER_SYSTEMS = "mean_over_systems"
    MEAN_OVER_MATRICES = "mean_over_matrices"
    MEDIAN = "median"


@dataclass(frozen=True, slots=True)
class MatrixClusterBootstrap:
    """The bootstrap ``matrix_cluster_interval`` runs, as a figure that draws its interval records it.

    ``level`` and ``resamples`` default to the constants the interval is computed with, so a
    freshly built record states the bootstrap actually used. ``matrices`` is the number of whole
    matrices resampled, the cluster count the interval rests on.
    """

    NAME: ClassVar[str] = "matrix_cluster_bootstrap"

    matrices: int
    level: float = CONFIDENCE_LEVEL
    resamples: int = BOOTSTRAP_RESAMPLES


def matrix_cluster_interval(
    values_by_matrix: Mapping[MatrixCluster, Sequence[float]],
    statistic: ClusterStatistic,
    *,
    seed: int,
) -> Interval | None:
    """Percentile interval of ``statistic`` resampling matrices with all their sibling values.

    ``None`` below two matrices. Equal values give a degenerate interval.

    Raises:
        ValueError: A matrix has no values.
    """
    if len(values_by_matrix) < MIN_INTERVAL_MATRICES:
        return None
    clusters = tuple(np.asarray(values, dtype=np.float64) for values in values_by_matrix.values())
    if any(cluster.size == 0 for cluster in clusters):
        raise ValueError("matrix clusters must contain at least one sample")
    all_values = np.concatenate(clusters)
    if np.all(all_values == all_values[0]):
        value = float(all_values[0])
        return Interval(value, value, CONFIDENCE_LEVEL)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(clusters), size=(BOOTSTRAP_RESAMPLES, len(clusters)))
    estimates = _resampled_statistic(clusters, picks, statistic)
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return Interval(float(low), float(high), CONFIDENCE_LEVEL)


def _resampled_statistic(
    clusters: tuple[np.ndarray, ...], picks: np.ndarray, statistic: ClusterStatistic
) -> np.ndarray:
    """The statistic of every resample; ``picks`` holds the cluster indices drawn per resample."""
    match statistic:
        case ClusterStatistic.MEAN_OVER_SYSTEMS:
            sums = np.asarray([cluster.sum() for cluster in clusters])
            counts = np.asarray([cluster.size for cluster in clusters])
            return sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
        case ClusterStatistic.MEAN_OVER_MATRICES:
            means = np.asarray([cluster.mean() for cluster in clusters])
            return means[picks].mean(axis=1)
        case ClusterStatistic.MEDIAN:
            return np.fromiter(
                (
                    np.median(np.concatenate(tuple(clusters[index] for index in selection)))
                    for selection in picks
                ),
                dtype=np.float64,
                count=len(picks),
            )
        case unreachable:
            assert_never(unreachable)
