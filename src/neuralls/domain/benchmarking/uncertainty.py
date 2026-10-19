"""Seeded percentile-bootstrap intervals over systems (Efron & Tibshirani 1993).

Systems are the resampling unit: runs on one system are not independent draws.
Every function is deterministic for a given seed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import bootstrap

from neuralls.domain.solver.models.statistics import Interval
from neuralls.shared.constants import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL

_MIN_SYSTEMS = 2


def median_interval(values: Sequence[float], *, seed: int) -> Interval | None:
    """Bootstrap interval of the median, or `None` below two values.

    Identical values give a zero-width interval at that value.
    """
    if len(values) < _MIN_SYSTEMS:
        return None
    if len(set(values)) == 1:
        return Interval(values[0], values[0], CONFIDENCE_LEVEL)
    result = bootstrap(
        (np.asarray(values, dtype=np.float64),),
        np.median,
        n_resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=CONFIDENCE_LEVEL,
        method="percentile",
        random_state=np.random.default_rng(seed),
    )
    interval = result.confidence_interval
    return Interval(float(interval.low), float(interval.high), CONFIDENCE_LEVEL)


def ert_interval(
    times: Sequence[float],
    converged: Sequence[bool],
    *,
    one_time_seconds: float,
    seed: int,
) -> Interval | None:
    """Bootstrap interval of ERT over systems.

    ERT is (one-time shared cost + recurring time of all runs) / number of
    successes. Resamples (recurring time, converged) pairs together and adds
    the constant ``one_time_seconds`` to each resample: the shared cost is
    paid once, whichever systems are drawn. Resamples without a success have
    no ERT and are discarded; `None` when fewer than two systems or every
    resample is discarded.
    """
    if len(times) < _MIN_SYSTEMS:
        return None
    costs = np.asarray(times, dtype=np.float64)
    successes = np.asarray(converged, dtype=np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(costs), size=(BOOTSTRAP_RESAMPLES, len(costs)))
    wins = successes[picks].sum(axis=1)
    keep = wins > 0
    if not keep.any():
        return None
    erts = (one_time_seconds + costs[picks].sum(axis=1)[keep]) / wins[keep]
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low, high = np.quantile(erts, [tail, 1.0 - tail])
    return Interval(float(low), float(high), CONFIDENCE_LEVEL)
