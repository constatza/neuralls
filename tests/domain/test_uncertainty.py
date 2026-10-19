"""Seeded bootstrap intervals (``domain.benchmarking.uncertainty``)."""

from __future__ import annotations

import math

import pytest

from neuralls.domain.benchmarking.uncertainty import ert_interval, median_interval

SEED = 3
VALUES = [float(v) for v in range(1, 11)]


def test_median_interval_brackets_the_median() -> None:
    interval = median_interval(VALUES, seed=SEED)

    assert interval is not None
    assert interval.low <= 5.5 <= interval.high
    assert interval.confidence == pytest.approx(0.95)


@pytest.mark.parametrize("values", [[], [1.0]])
def test_median_interval_is_none_below_two_values(values) -> None:
    assert median_interval(values, seed=SEED) is None


def test_identical_values_give_a_zero_width_interval() -> None:
    interval = median_interval([2.0, 2.0, 2.0], seed=SEED)

    assert interval is not None
    assert (interval.low, interval.high) == (2.0, 2.0)


def test_median_interval_is_deterministic_for_a_fixed_seed() -> None:
    assert median_interval(VALUES, seed=SEED) == median_interval(VALUES, seed=SEED)


def test_ert_interval_brackets_the_point_estimate() -> None:
    times = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0]
    converged = [True, True, True, False, True, True]

    interval = ert_interval(times, converged, one_time_seconds=0.0, seed=SEED)

    assert interval is not None
    assert interval.low <= sum(times) / 5 <= interval.high


def test_ert_interval_is_none_for_one_system_or_no_success() -> None:
    assert ert_interval([1.0], [True], one_time_seconds=0.0, seed=SEED) is None
    assert ert_interval([1.0, 2.0], [False, False], one_time_seconds=0.0, seed=SEED) is None


def test_resamples_without_a_success_are_discarded() -> None:
    # Equal times: a resample's ERT is 1 (both drawn systems) or 2 (one success
    # drawn twice with the failure); a zero-success resample would be infinite.
    interval = ert_interval([1.0, 1.0], [True, False], one_time_seconds=0.0, seed=SEED)

    assert interval is not None
    assert math.isfinite(interval.high)
    assert 1.0 <= interval.low <= interval.high <= 2.0


def test_ert_interval_is_deterministic_for_a_fixed_seed() -> None:
    args = ([1.0, 2.0, 3.0, 4.0], [True, False, True, True])

    assert ert_interval(*args, one_time_seconds=1.0, seed=SEED) == ert_interval(
        *args, one_time_seconds=1.0, seed=SEED
    )


def test_ert_interval_shifts_by_the_one_time_cost_per_success() -> None:
    times = [1.0] * 6
    converged = [True] * 6

    base = ert_interval(times, converged, one_time_seconds=0.0, seed=SEED)
    shared = ert_interval(times, converged, one_time_seconds=6.0, seed=SEED)

    assert base is not None and shared is not None
    # All six resampled systems always succeed: ERT = (one_time + 6) / 6.
    assert (base.low, base.high) == (1.0, 1.0)
    assert (shared.low, shared.high) == (2.0, 2.0)
