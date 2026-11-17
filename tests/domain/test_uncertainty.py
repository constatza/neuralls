"""Matrix-clustered uncertainty intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neuralls.domain.benchmarking.uncertainty import (
    MIN_INTERVAL_MATRICES,
    BootstrapKind,
    ClusterBootstrap,
    ClusterStatistic,
    TimedCluster,
    TooFewMatrices,
    cluster_bootstrap,
    matrix_cluster_interval,
    two_level_interval,
)
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


FIVE_MATRICES = {
    MatrixId(10): (1.0, 2.0, 3.0),
    MatrixId(20): (100.0,),
    MatrixId(30): (5.0,),
    MatrixId(40): (6.0,),
    MatrixId(50): (7.0,),
}


def test_matrix_cluster_interval_selects_every_sibling_together(monkeypatch) -> None:
    samples = FIVE_MATRICES

    class StubGenerator:
        def integers(self, low, high=None, size=None):
            del low, high
            return np.tile(np.zeros(len(samples), dtype=np.int64), (BOOTSTRAP_RESAMPLES, 1))

    monkeypatch.setattr(np.random, "default_rng", lambda seed: StubGenerator())

    interval = matrix_cluster_interval(samples, ClusterStatistic.MEDIAN, seed=SEED)

    assert interval is not None
    assert (interval.low, interval.high) == (2.0, 2.0)


def test_matrix_cluster_interval_differs_from_independent_rhs_resampling() -> None:
    clustered = matrix_cluster_interval(FIVE_MATRICES, ClusterStatistic.MEDIAN, seed=SEED)
    independent = _independent_rhs_interval(tuple(v for c in FIVE_MATRICES.values() for v in c))

    assert clustered is not None
    assert (clustered.low, clustered.high) != independent


def test_matrix_cluster_interval_is_none_below_the_minimum_matrices() -> None:
    too_few = dict(list(FIVE_MATRICES.items())[: MIN_INTERVAL_MATRICES - 1])

    assert matrix_cluster_interval(too_few, ClusterStatistic.MEDIAN, seed=SEED) is None


def test_matrix_cluster_interval_is_deterministic() -> None:
    assert matrix_cluster_interval(
        FIVE_MATRICES, ClusterStatistic.MEDIAN, seed=SEED
    ) == matrix_cluster_interval(FIVE_MATRICES, ClusterStatistic.MEDIAN, seed=SEED)


UNBALANCED = {
    MatrixId(10): (1.0, 2.0, 3.0),
    **{MatrixId(matrix): (10.0,) for matrix in (20, 30, 40, 50)},
}


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
        # picking cluster 0 twice and a single-value one thrice: sums (6, 6, 10, 10, 10)
        # over counts (3, 3, 1, 1, 1)
        (ClusterStatistic.MEAN_OVER_SYSTEMS, (0, 0, 1, 1, 1), 42.0 / 9.0),
        # cluster means (2, 2, 10, 10, 10), every drawn matrix weighs the same
        (ClusterStatistic.MEAN_OVER_MATRICES, (0, 0, 1, 1, 1), 34.0 / 5.0),
        # pooled values 1, 2, 3, 1, 2, 3, 10, 10, 10
        (ClusterStatistic.MEDIAN, (0, 0, 1, 1, 1), 3.0),
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
    samples = {MatrixId(matrix): (4.0, 4.0) for matrix in range(MIN_INTERVAL_MATRICES)}

    interval = matrix_cluster_interval(samples, statistic, seed=SEED)

    assert interval is not None
    assert (interval.low, interval.high, interval.confidence) == (4.0, 4.0, CONFIDENCE_LEVEL)


@pytest.mark.parametrize("statistic", list(ClusterStatistic))
def test_an_empty_cluster_is_rejected(statistic: ClusterStatistic) -> None:
    with pytest.raises(ValueError, match="at least one"):
        matrix_cluster_interval({**FIVE_MATRICES, MatrixId(60): ()}, statistic, seed=SEED)


@pytest.mark.parametrize("statistic", list(ClusterStatistic))
def test_the_interval_brackets_the_point_estimate_range(statistic: ClusterStatistic) -> None:
    interval = matrix_cluster_interval(UNBALANCED, statistic, seed=SEED)

    assert interval is not None
    assert 1.0 <= interval.low <= interval.high <= 10.0


def test_one_matrix_is_too_few_for_a_cluster_bootstrap() -> None:
    assert cluster_bootstrap(1, BootstrapKind.TWO_LEVEL) == TooFewMatrices(1)


def test_the_minimum_number_of_matrices_gets_a_cluster_bootstrap() -> None:
    assert cluster_bootstrap(MIN_INTERVAL_MATRICES, BootstrapKind.TWO_LEVEL) == ClusterBootstrap(
        kind=BootstrapKind.TWO_LEVEL, matrices=MIN_INTERVAL_MATRICES
    )


_TWO_LEVEL = BootstrapKind.TWO_LEVEL


@pytest.mark.parametrize(
    "build",
    [
        lambda: TooFewMatrices(-1),
        lambda: TooFewMatrices(MIN_INTERVAL_MATRICES),
        lambda: ClusterBootstrap(_TWO_LEVEL, matrices=1),
        lambda: ClusterBootstrap(_TWO_LEVEL, matrices=MIN_INTERVAL_MATRICES, level=1.0),
        lambda: ClusterBootstrap(_TWO_LEVEL, matrices=MIN_INTERVAL_MATRICES, resamples=0),
    ],
    ids=["negative", "enough", "bootstrap of one", "level of one", "no resamples"],
)
def test_an_interval_record_outside_its_range_is_rejected(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        build()


def _timed(values: tuple[float, ...], spread: float) -> TimedCluster:
    """One matrix whose values are each one solve timed three times around its value."""
    events = tuple((value * (1.0 - spread), value, value * (1.0 + spread)) for value in values)
    return TimedCluster(events, lambda times: times)


@pytest.mark.parametrize("statistic", list(ClusterStatistic))
@pytest.mark.parametrize("spread", [0.0, 0.5], ids=["no repeat spread", "repeat spread"])
def test_the_two_level_interval_adds_the_timing_repeats_to_the_matrix_one(
    statistic: ClusterStatistic, spread: float
) -> None:
    one_level = matrix_cluster_interval(UNBALANCED, statistic, seed=SEED)
    two_level = two_level_interval(
        {matrix: _timed(values, spread) for matrix, values in UNBALANCED.items()},
        statistic,
        seed=SEED,
    )

    assert one_level is not None
    assert two_level is not None
    if spread == 0.0:
        assert (two_level.low, two_level.high) == pytest.approx((one_level.low, one_level.high))
    else:
        assert two_level.high - two_level.low > one_level.high - one_level.low
