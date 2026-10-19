"""Paired comparison against a reference (``domain.benchmarking.paired``)."""

from __future__ import annotations

import pytest

from neuralls.domain.benchmarking.paired import PairedPoints, compare_paired, paired_points
from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome

SEED = 7


def _solved(times: list[float]) -> dict[str, SystemOutcome]:
    return {f"s{i}": Solved(1, t) for i, t in enumerate(times)}


@pytest.fixture
def reference() -> dict[str, SystemOutcome]:
    """s0, s1 solved; s2 failed; s3 failed; s4 solved with incomparable time; s5 solved."""
    return {
        "s0": Solved(1, 4.0),
        "s1": Solved(1, 6.0),
        "s2": Failed(),
        "s3": Failed(),
        "s4": Solved(1, None),
        "s5": Solved(1, 9.0),
    }


@pytest.fixture
def method() -> dict[str, SystemOutcome]:
    return {
        "s0": Solved(1, 2.0),
        "s1": Solved(1, 3.0),
        "s2": Solved(1, 5.0),
        "s3": Failed(),
        "s4": Solved(1, 1.0),
        "s5": Failed(),
    }


def test_points_split_pairs_method_only_and_reference_only(reference, method) -> None:
    points = paired_points(reference, method)

    assert points.both == ((4.0, 2.0), (6.0, 3.0))
    assert points.method_only == (5.0,)
    assert points.reference_only == (9.0,)


def test_incomparable_time_and_double_failure_are_skipped(reference, method) -> None:
    points = paired_points(reference, method)

    assert 1.0 not in points.method_only  # s4: reference solved without a time
    assert len(points.both) + len(points.method_only) + len(points.reference_only) == 4


def test_compare_paired_without_pairs_is_none() -> None:
    assert compare_paired(PairedPoints((), (1.0,), ()), seed=SEED) is None


def test_speedup_is_reference_over_method_time() -> None:
    result = compare_paired(PairedPoints(((4.0, 2.0), (6.0, 12.0)), (), ()), seed=SEED)

    assert result is not None
    assert result.speedups == (2.0, 0.5)
    assert result.median_speedup == pytest.approx(1.25)
    assert result.n_both_solved == 2


def test_wilcoxon_is_none_below_the_minimum_pair_count() -> None:
    points = PairedPoints(tuple((10.0 + i, 1.0) for i in range(5)), (), ())

    result = compare_paired(points, seed=SEED)

    assert result is not None
    assert result.wilcoxon_p is None


def test_wilcoxon_is_none_when_every_difference_is_zero() -> None:
    result = compare_paired(PairedPoints(tuple((2.0, 2.0) for _ in range(6)), (), ()), seed=SEED)

    assert result is not None
    assert result.wilcoxon_p is None


def test_wilcoxon_p_is_exact_for_six_pairs_all_in_one_direction() -> None:
    reference_times = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    method_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    points = PairedPoints(tuple(zip(reference_times, method_times, strict=True)), (), ())

    result = compare_paired(points, seed=SEED)

    assert result is not None
    assert result.wilcoxon_p == pytest.approx(2 / 2**6)  # two-sided exact: 2 / 64


def test_median_interval_is_deterministic_for_a_fixed_seed() -> None:
    points = PairedPoints(tuple((float(t), 1.0) for t in (2, 3, 5, 8, 13, 21)), (), ())

    first = compare_paired(points, seed=SEED)
    second = compare_paired(points, seed=SEED)

    assert first is not None
    assert first.median_speedup_ci is not None
    assert first == second


def test_wilcoxon_tests_log_ratios_not_raw_differences() -> None:
    """Five 10x speedups, one tiny slowdown and one tie: huge raw differences must not drive p."""
    pairs = tuple((10.0, 1.0) for _ in range(5)) + ((1.0, 1.2), (1.0, 1.0))

    result = compare_paired(PairedPoints(pairs, (), ()), seed=SEED)

    assert result is not None
    assert result.median_speedup == pytest.approx(10.0)
    # Signed-rank on log ratios: ranks of |log| are 1 (the 1.2 slowdown) and 2..6, so the
    # smaller signed sum is 1 and the two-sided exact p over 6 non-zero pairs is 4 / 64.
    assert result.wilcoxon_p == pytest.approx(4 / 2**6)


def test_non_positive_time_is_incomparable_not_an_infinite_speedup() -> None:
    reference = {"s0": Solved(1, 1.0), "s1": Solved(1, 2.0)}
    method = {"s0": Solved(1, 0.0), "s1": Solved(1, 1.0)}

    points = paired_points(reference, method)

    assert points.both == ((2.0, 1.0),)
