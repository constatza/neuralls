"""Paired frames and speed-ratio summaries over the observation view."""

from __future__ import annotations

import polars as pl
import pytest

from neuralls.domain.benchmarking.paired import (
    PAIRED_SCHEMA,
    SPEED_RATIO_SCHEMA,
    PairKind,
    paired_systems,
    speed_ratio_summary,
    speed_ratios,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.shared.types import PreconditionerName
from tests.observation_support import AMG, BROKEN, JACOBI

SEED = 7

type SummaryCounts = tuple[int, int, int, int, int]
"""n_systems_both_solved, n_ratio, n_matrices, n_reference_only, n_method_only."""

type RatioStatistics = tuple[float, float, float]
"""median_ratio, ci_low, ci_high."""

# amortized online time = solve + setup / planned RHS of the matrix:
# amg 1+6/3, 2+6/3, 3+6/3 (matrix 1), 4+12/3 (matrix 2); jacobi 0.5+1/3, 0.6+1/3, 0.7+1/3,
# 0.8+2/3. The ratio is reference over method; the matrix-cluster interval of a median over two
# matrices spans the medians of matrix 1 alone, of matrix 2 alone and of both pooled.
REFERENCE_COMPARISONS: tuple[
    tuple[
        PreconditionerName, PreconditionerName, list[float], SummaryCounts, RatioStatistics | None
    ],
    ...,
] = (
    (
        JACOBI,
        AMG,
        [(0.5 + 1 / 3) / 3, (0.6 + 1 / 3) / 4, (0.7 + 1 / 3) / 5, (0.8 + 2 / 3) / 8],
        (4, 4, 2, 5, 0),
        (0.22, (0.8 + 2 / 3) / 8, (0.6 + 1 / 3) / 4),
    ),
    (
        AMG,
        JACOBI,
        [3 / (0.5 + 1 / 3), 4 / (0.6 + 1 / 3), 5 / (0.7 + 1 / 3), 8 / (0.8 + 2 / 3)],
        (4, 4, 2, 0, 5),
        (
            (4 / (0.6 + 1 / 3) + 5 / (0.7 + 1 / 3)) / 2,
            4 / (0.6 + 1 / 3),
            8 / (0.8 + 2 / 3),
        ),
    ),
    (JACOBI, BROKEN, [], (0, 0, 0, 9, 0), None),
    (AMG, BROKEN, [], (0, 0, 0, 4, 0), None),
)
"""Reference, method, the ratios, the counts and the statistics of each pairing."""


def _summary_row(summary: pl.DataFrame, method: str) -> dict[str, object]:
    return summary.filter(pl.col("method") == method).row(0, named=True)


@pytest.mark.parametrize(
    ("reference", "method", "ratios", "counts", "statistics"), REFERENCE_COMPARISONS
)
def test_speed_ratio_summary_of_the_online_time_of_each_pairing(
    unbalanced_observations: pl.DataFrame,
    reference: PreconditionerName,
    method: PreconditionerName,
    ratios: list[float],
    counts: SummaryCounts,
    statistics: RatioStatistics | None,
) -> None:
    per_system = speed_ratios(paired_systems(unbalanced_observations, reference, MetricId.ONLINE_S))
    row = _summary_row(
        speed_ratio_summary(unbalanced_observations, reference, MetricId.ONLINE_S, seed=SEED),
        method,
    )

    assert sorted(per_system.filter(pl.col("method") == method)["ratio"]) == pytest.approx(
        sorted(ratios)
    )
    assert (
        tuple(
            row[key]
            for key in (
                "n_systems_both_solved",
                "n_ratio",
                "n_matrices",
                "n_reference_only",
                "n_method_only",
            )
        )
        == counts
    )
    observed = (row["median_ratio"], row["ci_low"], row["ci_high"])
    if statistics is None:
        assert observed == (None, None, None)
    else:
        assert observed == pytest.approx(statistics)


def test_paired_frame_has_the_declared_schema_and_never_the_reference_as_method(
    unbalanced_observations: pl.DataFrame,
) -> None:
    paired = paired_systems(unbalanced_observations, JACOBI, MetricId.SOLVE_S)

    assert paired.schema == pl.Schema(PAIRED_SCHEMA)
    assert set(paired["method"]) == {AMG, BROKEN}


def test_pair_kinds_hold_only_the_values_that_exist(
    unbalanced_observations: pl.DataFrame,
) -> None:
    paired = paired_systems(unbalanced_observations, JACOBI, MetricId.SOLVE_S)
    amg = paired.filter(pl.col("method") == AMG)

    assert amg.select("system_id", "pair_kind", "reference_value", "method_value").rows() == [
        ("m1-0", PairKind.BOTH, 0.5, 1.0),
        ("m1-1", PairKind.BOTH, 0.6, 2.0),
        ("m1-2", PairKind.BOTH, 0.7, 3.0),
        ("m2-0", PairKind.BOTH, 0.8, 4.0),
        ("m2-1", PairKind.REFERENCE_ONLY, 0.9, None),
        ("m2-2", PairKind.REFERENCE_ONLY, 1.0, None),
        ("m3-0", PairKind.REFERENCE_ONLY, 0.1, None),
        ("m3-1", PairKind.REFERENCE_ONLY, 0.2, None),
        ("m4-0", PairKind.REFERENCE_ONLY, 0.3, None),
    ]


def test_method_only_systems_carry_no_reference_value(
    unbalanced_observations: pl.DataFrame,
) -> None:
    paired = paired_systems(unbalanced_observations, AMG, MetricId.SOLVE_S)
    method_only = paired.filter(
        (pl.col("method") == JACOBI) & (pl.col("pair_kind") == PairKind.METHOD_ONLY)
    )

    assert method_only["system_id"].to_list() == ["m2-1", "m2-2", "m3-0", "m3-1", "m4-0"]
    assert method_only["reference_value"].null_count() == method_only.height
    assert method_only["method_value"].null_count() == 0


def test_systems_neither_side_solved_are_not_in_the_frame(
    unbalanced_observations: pl.DataFrame,
) -> None:
    paired = paired_systems(unbalanced_observations, AMG, MetricId.SOLVE_S)
    broken = paired.filter(pl.col("method") == BROKEN)

    assert broken["system_id"].to_list() == ["m1-0", "m1-1", "m1-2", "m2-0"]
    assert set(broken["pair_kind"]) == {PairKind.REFERENCE_ONLY}


def test_a_reference_absent_from_the_observations_is_rejected(
    two_method_observations: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="reference"):
        paired_systems(two_method_observations, BROKEN, MetricId.SOLVE_S)
    with pytest.raises(ValueError, match="reference"):
        speed_ratio_summary(two_method_observations, BROKEN, MetricId.SOLVE_S)


def test_a_matrix_unit_metric_is_rejected(two_method_observations: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="setup_s"):
        paired_systems(two_method_observations, JACOBI, MetricId.SETUP_S)


def test_summary_counts_the_distinct_matrices_the_interval_rests_on(
    two_method_observations: pl.DataFrame,
) -> None:
    summary = speed_ratio_summary(two_method_observations, JACOBI, MetricId.SOLVE_S, seed=SEED)

    assert summary.schema == pl.Schema(SPEED_RATIO_SCHEMA)
    assert summary.drop("median_ratio", "ci_low", "ci_high").row(0, named=True) == {
        "method": AMG,
        "n_systems_both_solved": 4,
        "n_ratio": 4,
        "n_matrices": 2,
        "n_reference_only": 5,
        "n_method_only": 0,
    }


def test_the_median_is_that_of_the_reference_over_method_ratios(
    two_method_observations: pl.DataFrame,
) -> None:
    summary = speed_ratio_summary(two_method_observations, JACOBI, MetricId.SOLVE_S, seed=SEED)

    # 0.5/1, 0.6/2, 0.7/3, 0.8/4 = 0.5, 0.3, 0.2333.., 0.2: median (0.2333.. + 0.3) / 2
    assert _summary_row(summary, AMG)["median_ratio"] == pytest.approx((0.7 / 3 + 0.3) / 2)


def test_a_zero_time_is_counted_as_both_solved_but_gives_no_ratio(
    zero_solve_time_observations: pl.DataFrame,
) -> None:
    """``amg`` takes 0 s on the three systems of matrix 1: reference over 0 is undefined."""
    row = _summary_row(
        speed_ratio_summary(zero_solve_time_observations, JACOBI, MetricId.SOLVE_S, seed=SEED),
        AMG,
    )

    assert row["n_systems_both_solved"] == 4
    assert row["n_ratio"] == 1
    assert row["n_matrices"] == 1
    assert row["median_ratio"] == pytest.approx(0.8 / 4)
    assert (row["ci_low"], row["ci_high"]) == (None, None)


def test_a_method_with_no_ratio_has_null_statistics_but_its_counts(
    unbalanced_observations: pl.DataFrame,
) -> None:
    row = _summary_row(
        speed_ratio_summary(unbalanced_observations, JACOBI, MetricId.SOLVE_S, seed=SEED), BROKEN
    )

    assert row == {
        "method": BROKEN,
        "n_systems_both_solved": 0,
        "n_ratio": 0,
        "n_matrices": 0,
        "n_reference_only": 9,
        "n_method_only": 0,
        "median_ratio": None,
        "ci_low": None,
        "ci_high": None,
    }
