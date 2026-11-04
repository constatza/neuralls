"""Matrix-clustered paired comparisons against a reference, from hand-written observations."""

from __future__ import annotations

import polars as pl
import pytest

from neuralls.domain.benchmarking.paired import (
    PairKind,
    paired_systems,
    speed_ratio_summary,
    speed_ratios,
)
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.shared.types import DatasetId, PreconditionerName
from tests.observation_support import (
    SyntheticRow,
    solved_row,
    synthetic_observations,
    unconverged_row,
)

SEED = 7
REFERENCE = PreconditionerName("reference")
METHOD = PreconditionerName("method")


def _summary(rows: list[SyntheticRow]) -> dict[str, object]:
    obs = synthetic_observations(rows)
    return speed_ratio_summary(obs, REFERENCE, MetricId.ONLINE_S, seed=SEED).row(0, named=True)


def _counts(summary: dict[str, object]) -> tuple[object, object, object]:
    return (
        summary["n_systems_both_solved"],
        summary["n_reference_only"],
        summary["n_method_only"],
    )


@pytest.fixture
def mixed_rows() -> list[SyntheticRow]:
    """s0 and s1 solved by both, s2 only by the method, s3 by neither, s4 only by the reference."""
    return [
        solved_row(REFERENCE, "s0", 10, 4.0),
        solved_row(REFERENCE, "s1", 10, 6.0),
        unconverged_row(REFERENCE, "s2", 20),
        unconverged_row(REFERENCE, "s3", 20),
        solved_row(REFERENCE, "s4", 30, 9.0),
        solved_row(METHOD, "s0", 10, 2.0),
        solved_row(METHOD, "s1", 10, 3.0),
        solved_row(METHOD, "s2", 20, 5.0),
        unconverged_row(METHOD, "s3", 20),
        unconverged_row(METHOD, "s4", 30),
    ]


@pytest.fixture
def three_ratios_on_one_matrix_and_one_on_another() -> list[SyntheticRow]:
    """Ratios 2, 4, 6 on matrix 10 and 100 on matrix 20."""
    reference = {"a0": (10, 2.0), "a1": (10, 4.0), "a2": (10, 6.0), "b0": (20, 100.0)}
    return [
        row
        for system, (matrix, seconds) in reference.items()
        for row in (
            solved_row(REFERENCE, system, matrix, seconds),
            solved_row(METHOD, system, matrix, 1.0),
        )
    ]


def test_pairs_split_into_both_method_only_and_reference_only(
    mixed_rows: list[SyntheticRow],
) -> None:
    paired = paired_systems(synthetic_observations(mixed_rows), REFERENCE, MetricId.ONLINE_S)

    assert paired.select("system_id", "pair_kind", "reference_value", "method_value").rows() == [
        ("s0", PairKind.BOTH, 4.0, 2.0),
        ("s1", PairKind.BOTH, 6.0, 3.0),
        ("s2", PairKind.METHOD_ONLY, None, 5.0),
        ("s4", PairKind.REFERENCE_ONLY, 9.0, None),
    ]


def test_speed_ratio_fields_and_asymmetric_counts(mixed_rows: list[SyntheticRow]) -> None:
    obs = synthetic_observations(mixed_rows)
    summary = _summary(mixed_rows)

    assert speed_ratios(paired_systems(obs, REFERENCE, MetricId.ONLINE_S))["ratio"].to_list() == [
        2.0,
        2.0,
    ]
    assert summary["median_ratio"] == pytest.approx(2.0)
    assert _counts(summary) == (2, 1, 1)


def test_asymmetric_failures_survive_without_any_pair() -> None:
    summary = _summary([unconverged_row(REFERENCE, "s0", 1), solved_row(METHOD, "s0", 1, 1.0)])

    assert summary["median_ratio"] is None
    assert summary["n_ratio"] == 0
    assert summary["n_method_only"] == 1


def test_ratio_interval_resamples_matrix_clusters(
    three_ratios_on_one_matrix_and_one_on_another: list[SyntheticRow],
) -> None:
    summary = _summary(three_ratios_on_one_matrix_and_one_on_another)

    assert summary["n_matrices"] == 2
    assert summary["ci_low"] is not None
    assert summary == _summary(three_ratios_on_one_matrix_and_one_on_another)


def test_ratio_interval_clusters_same_matrix_id_separately_by_dataset() -> None:
    rows = [
        solved_row(REFERENCE, "a", 1, 2.0, dataset=DatasetId("ds-a")),
        solved_row(REFERENCE, "b", 1, 4.0, dataset=DatasetId("ds-b")),
        solved_row(METHOD, "a", 1, 1.0, dataset=DatasetId("ds-a")),
        solved_row(METHOD, "b", 1, 1.0, dataset=DatasetId("ds-b")),
    ]

    summary = _summary(rows)

    assert summary["n_matrices"] == 2
    assert summary["ci_low"] is not None


def test_non_positive_time_is_not_turned_into_infinite_ratio() -> None:
    rows = [
        solved_row(REFERENCE, "s0", 1, 1.0),
        solved_row(REFERENCE, "s1", 1, 2.0),
        solved_row(METHOD, "s0", 1, 0.0),
        solved_row(METHOD, "s1", 1, 1.0),
    ]
    obs = synthetic_observations(rows)

    ratios = speed_ratios(paired_systems(obs, REFERENCE, MetricId.ONLINE_S))["ratio"].to_list()

    assert ratios == [2.0]


def test_zero_time_both_solved_counts_both_without_ratio() -> None:
    summary = _summary([solved_row(REFERENCE, "s0", 1, 1.0), solved_row(METHOD, "s0", 1, 0.0)])

    assert _counts(summary) == (1, 0, 0)
    assert summary["n_ratio"] == 0


def test_zero_time_reference_only_counts_reference_success() -> None:
    summary = _summary([solved_row(REFERENCE, "s0", 1, 0.0), unconverged_row(METHOD, "s0", 1)])

    assert _counts(summary) == (0, 1, 0)


def test_zero_time_method_only_counts_method_success() -> None:
    summary = _summary([unconverged_row(REFERENCE, "s0", 1), solved_row(METHOD, "s0", 1, 0.0)])

    assert _counts(summary) == (0, 0, 1)


def test_speed_ratios_keep_the_identity_of_the_system_each_ratio_belongs_to(
    mixed_rows: list[SyntheticRow],
) -> None:
    paired = paired_systems(synthetic_observations(mixed_rows), REFERENCE, MetricId.ONLINE_S)

    assert speed_ratios(paired).select(
        "method", "dataset", "matrix_id", "system_id", "ratio"
    ).rows() == [
        (METHOD, "ds", 10, "s0", 2.0),
        (METHOD, "ds", 10, "s1", 2.0),
    ]
    assert speed_ratios(paired).schema == pl.Schema(
        {
            "method": pl.String,
            "dataset": pl.String,
            "matrix_id": pl.Int64,
            "system_id": pl.String,
            "ratio": pl.Float64,
        }
    )
