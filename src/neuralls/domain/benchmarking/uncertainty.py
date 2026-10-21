"""Seeded percentile-bootstrap intervals over whole matrix clusters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from neuralls.domain.solver.models.statistics import Interval
from neuralls.shared.constants import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
from neuralls.shared.types import MatrixId

type MatrixCluster = MatrixId | tuple[str | None, MatrixId]

_MIN_MATRICES = 2


def matrix_cluster_median_interval(
    samples_by_matrix: Mapping[MatrixCluster, Sequence[float]], *, seed: int
) -> Interval | None:
    """Median interval resampling matrices with all their sibling RHS values."""
    if len(samples_by_matrix) < _MIN_MATRICES:
        return None
    clusters = tuple(np.asarray(values, dtype=np.float64) for values in samples_by_matrix.values())
    if any(cluster.size == 0 for cluster in clusters):
        raise ValueError("matrix clusters must contain at least one sample")
    all_values = np.concatenate(clusters)
    if np.all(all_values == all_values[0]):
        value = float(all_values[0])
        return Interval(value, value, CONFIDENCE_LEVEL)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(clusters), size=(BOOTSTRAP_RESAMPLES, len(clusters)))
    medians = np.fromiter(
        (
            np.median(np.concatenate(tuple(clusters[index] for index in selection)))
            for selection in picks
        ),
        dtype=np.float64,
        count=BOOTSTRAP_RESAMPLES,
    )
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low, high = np.quantile(medians, [tail, 1.0 - tail])
    return Interval(float(low), float(high), CONFIDENCE_LEVEL)
