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
    return speed_ratio_summary(obs, REFERENCE, MetricId.ONLINE_AMORTIZED_S, seed=SEED).row(
        0, named=True
    )


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
def three_ratios_on_one_matrix_and_one_on_others() -> list[SyntheticRow]:
    """Ratios 2, 4, 6 on matrix 10 and 100, 3, 5, 7 on matrices 20 to 50."""
    reference = {
        "a0": (10, 2.0),
        "a1": (10, 4.0),
        "a2": (10, 6.0),
        "b0": (20, 100.0),
        "c0": (30, 3.0),
        "d0": (40, 5.0),
        "e0": (50, 7.0),
    }
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
    paired = paired_systems(
        synthetic_observations(mixed_rows), REFERENCE, MetricId.ONLINE_AMORTIZED_S
    )

    assert paired.select("system_id", "pair_kind", "reference_value", "method_value").rows() == [
        ("s0", PairKind.BOTH, 4.0, 2.0),
        ("s1", PairKind.BOTH, 6.0, 3.0),
        ("s2", PairKind.METHOD_ONLY, None, 5.0),
        ("s4", PairKind.REFERENCE_ONLY, 9.0, None),
    ]


def test_speed_ratio_fields_and_asymmetric_counts(mixed_rows: list[SyntheticRow]) -> None:
    obs = synthetic_observations(mixed_rows)
    summary = _summary(mixed_rows)

    assert speed_ratios(paired_systems(obs, REFERENCE, MetricId.ONLINE_AMORTIZED_S))[
        "ratio"
    ].to_list() == [
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
    three_ratios_on_one_matrix_and_one_on_others: list[SyntheticRow],
) -> None:
    summary = _summary(three_ratios_on_one_matrix_and_one_on_others)

    assert summary["n_matrices"] == 5
    assert summary["ci_low"] is not None
    assert summary == _summary(three_ratios_on_one_matrix_and_one_on_others)


def test_ratio_interval_clusters_same_matrix_id_separately_by_dataset() -> None:
    reference_seconds = {"a": 2.0, "b": 4.0, "c": 3.0, "d": 5.0, "e": 6.0}
    rows = [
        row
        for name, seconds in reference_seconds.items()
        for row in (
            solved_row(REFERENCE, name, 1, seconds, dataset=DatasetId(f"ds-{name}")),
            solved_row(METHOD, name, 1, 1.0, dataset=DatasetId(f"ds-{name}")),
        )
    ]

    summary = _summary(rows)

    assert summary["n_matrices"] == len(reference_seconds)
    assert summary["ci_low"] is not None


def test_non_positive_time_is_not_turned_into_infinite_ratio() -> None:
    rows = [
        solved_row(REFERENCE, "s0", 1, 1.0),
        solved_row(REFERENCE, "s1", 1, 2.0),
        solved_row(METHOD, "s0", 1, 0.0),
        solved_row(METHOD, "s1", 1, 1.0),
    ]
    obs = synthetic_observations(rows)

    ratios = speed_ratios(paired_systems(obs, REFERENCE, MetricId.ONLINE_AMORTIZED_S))[
        "ratio"
    ].to_list()

    assert ratios == [2.0]


@pytest.mark.parametrize(
    ("rows", "counts"),
    [
        (
            [solved_row(REFERENCE, "s0", 1, 1.0), solved_row(METHOD, "s0", 1, 0.0)],
            (1, 0, 0),
        ),
        (
            [solved_row(REFERENCE, "s0", 1, 0.0), unconverged_row(METHOD, "s0", 1)],
            (0, 1, 0),
        ),
        (
            [unconverged_row(REFERENCE, "s0", 1), solved_row(METHOD, "s0", 1, 0.0)],
            (0, 0, 1),
        ),
    ],
    ids=["both_solved", "reference_only", "method_only"],
)
def test_a_zero_time_counts_its_success_without_a_ratio(
    rows: list[SyntheticRow], counts: tuple[int, int, int]
) -> None:
    summary = _summary(rows)

    assert _counts(summary) == counts
    assert summary["n_ratio"] == 0


def test_speed_ratios_keep_the_identity_of_the_system_each_ratio_belongs_to(
    mixed_rows: list[SyntheticRow],
) -> None:
    paired = paired_systems(
        synthetic_observations(mixed_rows), REFERENCE, MetricId.ONLINE_AMORTIZED_S
    )

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
