"""Matrix-clustered uncertainty intervals."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.benchmarking.uncertainty import ClusterStatistic, matrix_cluster_interval
from neuralls.shared.constants import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
from neuralls.shared.types import MatrixId

SEED = 187


def _independent_rhs_interval(values: tuple[float, ...]) -> tuple[float, float]:
    """Hand-written row bootstrap used only to prove clustering changes the answer."""
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    estimates = np.median(np.asarray(values)[picks], axis=1)
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low, high = np.quantile(estimates, (tail, 1.0 - tail))
    return float(low), float(high)


def test_matrix_cluster_interval_selects_every_sibling_together(monkeypatch) -> None:
    samples = {MatrixId(10): (1.0, 2.0, 3.0), MatrixId(20): (100.0,)}

    class StubGenerator:
        def integers(self, low, high=None, size=None):
            del low, high
            return np.tile(np.asarray((0, 0)), (BOOTSTRAP_RESAMPLES, 1))

    monkeypatch.setattr(np.random, "default_rng", lambda seed: StubGenerator())

    interval = matrix_cluster_interval(samples, ClusterStatistic.MEDIAN, seed=SEED)

    assert interval is not None
    assert (interval.low, interval.high) == (2.0, 2.0)


def test_matrix_cluster_interval_differs_from_independent_rhs_resampling() -> None:
    samples = {MatrixId(10): (1.0, 2.0, 3.0), MatrixId(20): (100.0,)}

    clustered = matrix_cluster_interval(samples, ClusterStatistic.MEDIAN, seed=SEED)
    independent = _independent_rhs_interval((1.0, 2.0, 3.0, 100.0))

    assert clustered is not None
    assert (clustered.low, clustered.high) != independent


def test_matrix_cluster_interval_is_none_below_two_matrices() -> None:
    assert (
        matrix_cluster_interval({MatrixId(10): (1.0, 2.0)}, ClusterStatistic.MEDIAN, seed=SEED)
        is None
    )


def test_matrix_cluster_interval_is_deterministic() -> None:
    samples = {MatrixId(10): (1.0, 2.0), MatrixId(20): (3.0,), MatrixId(30): (4.0, 5.0)}

    assert matrix_cluster_interval(
        samples, ClusterStatistic.MEDIAN, seed=SEED
    ) == matrix_cluster_interval(samples, ClusterStatistic.MEDIAN, seed=SEED)


UNBALANCED = {MatrixId(10): (1.0, 2.0, 3.0), MatrixId(20): (10.0,)}


def _picks(selection: tuple[int, ...]) -> np.ndarray:
    return np.tile(np.asarray(selection), (BOOTSTRAP_RESAMPLES, 1))


class _FixedPicks:
    """Generator stub that makes every resample draw the same clusters."""

    def __init__(self, selection: tuple[int, ...]) -> None:
        self._selection = selection

    def integers(self, low, high=None, size=None):
        del low, high, size
        return _picks(self._selection)


@pytest.mark.parametrize(
    ("statistic", "selection", "expected"),
    [
        # picking cluster 0 twice and cluster 1 once: sums (6, 6, 10) over counts (3, 3, 1)
        (ClusterStatistic.MEAN_OVER_SYSTEMS, (0, 0, 1), 22.0 / 7.0),
        # cluster means (2, 2, 10), every drawn matrix weighs the same
        (ClusterStatistic.MEAN_OVER_MATRICES, (0, 0, 1), 14.0 / 3.0),
        # pooled values 1, 2, 3, 1, 2, 3, 10
        (ClusterStatistic.MEDIAN, (0, 0, 1), 2.0),
    ],
)
def test_each_statistic_is_recomputed_on_the_drawn_clusters(
    monkeypatch, statistic: ClusterStatistic, selection: tuple[int, ...], expected: float
) -> None:
    monkeypatch.setattr(np.random, "default_rng", lambda seed: _FixedPicks(selection))

    interval = matrix_cluster_interval(UNBALANCED, statistic, seed=SEED)

    assert interval is not None
    assert (interval.low, interval.high) == pytest.approx((expected, expected))


@pytest.mark.parametrize("statistic", list(ClusterStatistic))
def test_equal_values_give_a_degenerate_interval(statistic: ClusterStatistic) -> None:
    samples = {MatrixId(10): (4.0, 4.0), MatrixId(20): (4.0,)}

    interval = matrix_cluster_interval(samples, statistic, seed=SEED)

    assert interval is not None
    assert (interval.low, interval.high, interval.confidence) == (4.0, 4.0, CONFIDENCE_LEVEL)


@pytest.mark.parametrize("statistic", list(ClusterStatistic))
def test_an_empty_cluster_is_rejected(statistic: ClusterStatistic) -> None:
    with pytest.raises(ValueError, match="at least one"):
        matrix_cluster_interval({MatrixId(10): (1.0,), MatrixId(20): ()}, statistic, seed=SEED)


@pytest.mark.parametrize("statistic", list(ClusterStatistic))
def test_the_interval_brackets_the_point_estimate_range(statistic: ClusterStatistic) -> None:
    interval = matrix_cluster_interval(UNBALANCED, statistic, seed=SEED)

    assert interval is not None
    assert 1.0 <= interval.low <= interval.high <= 10.0
