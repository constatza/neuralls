"""Matrix-clustered uncertainty intervals."""

from __future__ import annotations

import numpy as np

from neuralls.domain.benchmarking.uncertainty import matrix_cluster_median_interval
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

    interval = matrix_cluster_median_interval(samples, seed=SEED)

    assert interval is not None
    assert (interval.low, interval.high) == (2.0, 2.0)


def test_matrix_cluster_interval_differs_from_independent_rhs_resampling() -> None:
    samples = {MatrixId(10): (1.0, 2.0, 3.0), MatrixId(20): (100.0,)}

    clustered = matrix_cluster_median_interval(samples, seed=SEED)
    independent = _independent_rhs_interval((1.0, 2.0, 3.0, 100.0))

    assert clustered is not None
    assert (clustered.low, clustered.high) != independent


def test_matrix_cluster_interval_is_none_below_two_matrices() -> None:
    assert matrix_cluster_median_interval({MatrixId(10): (1.0, 2.0)}, seed=SEED) is None


def test_matrix_cluster_interval_is_deterministic() -> None:
    samples = {MatrixId(10): (1.0, 2.0), MatrixId(20): (3.0,), MatrixId(30): (4.0, 5.0)}

    assert matrix_cluster_median_interval(samples, seed=SEED) == matrix_cluster_median_interval(
        samples, seed=SEED
    )
