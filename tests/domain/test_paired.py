"""Matrix-clustered paired comparisons against a reference."""

from __future__ import annotations

import pytest

from neuralls.domain.benchmarking.paired import compare_paired, paired_points
from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome
from neuralls.shared.types import MatrixId

SEED = 7


def _solved(matrix: int, time: float, iterations: int = 1) -> Solved:
    return Solved(MatrixId(matrix), iterations, solve_s=time, amortized_online_s=time)


@pytest.fixture
def reference() -> dict[str, SystemOutcome]:
    return {
        "s0": _solved(10, 4.0),
        "s1": _solved(10, 6.0),
        "s2": Failed(MatrixId(20), 100, "limit"),
        "s3": Failed(MatrixId(20), 3, "breakdown"),
        "s4": _solved(30, 9.0),
    }


@pytest.fixture
def method() -> dict[str, SystemOutcome]:
    return {
        "s0": _solved(10, 2.0),
        "s1": _solved(10, 3.0),
        "s2": _solved(20, 5.0),
        "s3": Failed(MatrixId(20), 3, "breakdown"),
        "s4": Failed(MatrixId(30), 100, "limit"),
    }


def test_points_split_pairs_method_only_and_reference_only(reference, method) -> None:
    points = paired_points(reference, method)

    assert tuple(
        (ref.amortized_online_s, candidate.amortized_online_s) for ref, candidate in points.both
    ) == (
        (4.0, 2.0),
        (6.0, 3.0),
    )
    assert tuple(point.amortized_online_s for point in points.method_only) == (5.0,)
    assert tuple(point.amortized_online_s for point in points.reference_only) == (9.0,)


def test_speed_ratio_fields_and_asymmetric_counts(reference, method) -> None:
    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert result.speed_ratios == (2.0, 2.0)
    assert result.median_speed_ratio == pytest.approx(2.0)
    assert (result.n_both_solved, result.n_reference_only, result.n_method_only) == (2, 1, 1)


def test_asymmetric_failures_survive_without_any_pair() -> None:
    reference = {"s0": Failed(MatrixId(1), 10, "limit")}
    method = {"s0": _solved(1, 1.0)}

    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert result.speed_ratios == ()
    assert result.median_speed_ratio is None
    assert result.n_method_only == 1


def test_ratio_interval_resamples_matrix_clusters() -> None:
    reference = {
        "a0": _solved(10, 2.0),
        "a1": _solved(10, 4.0),
        "a2": _solved(10, 6.0),
        "b0": _solved(20, 100.0),
    }
    method = {
        system_id: _solved(outcome.matrix_id, 1.0) for system_id, outcome in reference.items()
    }

    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert result.median_speed_ratio_ci is not None
    assert result == compare_paired(paired_points(reference, method), seed=SEED)


def test_ratio_interval_clusters_same_matrix_id_separately_by_dataset() -> None:
    reference = {
        "a": Solved(MatrixId(1), 1, 2.0, 2.0, dataset="ds-a"),
        "b": Solved(MatrixId(1), 1, 4.0, 4.0, dataset="ds-b"),
    }
    method = {
        "a": Solved(MatrixId(1), 1, 1.0, 1.0, dataset="ds-a"),
        "b": Solved(MatrixId(1), 1, 1.0, 1.0, dataset="ds-b"),
    }

    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert result.median_speed_ratio_ci is not None


def test_wilcoxon_is_none_below_minimum_pair_count(reference, method) -> None:
    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert not hasattr(result, "wilcoxon_p")


def test_sibling_rhs_duplication_does_not_create_wilcoxon_evidence() -> None:
    reference = {f"s{i}": _solved(i // 2, 10.0 + 2 * i) for i in range(6)}
    method = {f"s{i}": _solved(i // 2, 1.0 + i) for i in range(6)}

    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert not hasattr(result, "wilcoxon_p")


def test_non_positive_time_is_not_turned_into_infinite_ratio() -> None:
    reference = {"s0": _solved(1, 1.0), "s1": _solved(1, 2.0)}
    method = {"s0": _solved(1, 0.0), "s1": _solved(1, 1.0)}

    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert result.speed_ratios == (2.0,)


def test_zero_time_both_solved_counts_both_without_ratio() -> None:
    reference = {"s0": _solved(1, 1.0)}
    method = {"s0": _solved(1, 0.0)}

    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert (result.n_both_solved, result.n_reference_only, result.n_method_only) == (1, 0, 0)
    assert result.speed_ratios == ()


def test_zero_time_reference_only_counts_reference_success() -> None:
    reference = {"s0": _solved(1, 0.0)}
    method = {"s0": Failed(MatrixId(1), 1, "limit")}

    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert (result.n_both_solved, result.n_reference_only, result.n_method_only) == (0, 1, 0)


def test_zero_time_method_only_counts_method_success() -> None:
    reference = {"s0": Failed(MatrixId(1), 1, "limit")}
    method = {"s0": _solved(1, 0.0)}

    result = compare_paired(paired_points(reference, method), seed=SEED)

    assert (result.n_both_solved, result.n_reference_only, result.n_method_only) == (0, 0, 1)
