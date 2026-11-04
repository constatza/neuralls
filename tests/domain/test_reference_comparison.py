"""Explicit reference comparison."""

from __future__ import annotations

import polars as pl
import pytest

from neuralls.domain.benchmarking.paired import (
    REFERENCE_METRIC,
    paired_systems,
    reference_among_methods,
    speed_ratio_summary,
    speed_ratios,
)
from neuralls.domain.benchmarking.selection import ConfiguredReference, NoReferenceConfigured
from neuralls.shared.types import PreconditionerName
from tests.observation_support import solved_row, synthetic_observations, unconverged_row

SEED = 11
AMG = PreconditionerName("amg")
JACOBI = PreconditionerName("jacobi")
NEVER = PreconditionerName("never")


@pytest.fixture
def observations() -> pl.DataFrame:
    """``amg`` solves all three systems, ``jacobi`` the first two, ``never`` none."""
    return synthetic_observations(
        [
            solved_row(AMG, "s0", 1, 4.0),
            solved_row(AMG, "s1", 1, 6.0),
            solved_row(AMG, "s2", 1, 8.0),
            solved_row(JACOBI, "s0", 1, 2.0),
            solved_row(JACOBI, "s1", 1, 3.0),
            unconverged_row(JACOBI, "s2", 1),
            unconverged_row(NEVER, "s0", 1),
            unconverged_row(NEVER, "s1", 1),
            unconverged_row(NEVER, "s2", 1),
        ]
    )


def test_reference_is_skipped_but_every_other_method_is_compared(
    observations: pl.DataFrame,
) -> None:
    summary = speed_ratio_summary(observations, AMG, REFERENCE_METRIC, seed=SEED)

    assert summary["method"].to_list() == [JACOBI, NEVER]
    assert summary.filter(pl.col("method") == NEVER)["n_reference_only"].to_list() == [3]


def test_paired_part_uses_reference_over_method_ratio(observations: pl.DataFrame) -> None:
    ratios = speed_ratios(paired_systems(observations, AMG, REFERENCE_METRIC))

    assert ratios.filter(pl.col("method") == JACOBI)["ratio"].to_list() == [2.0, 2.0]


def test_a_configured_reference_among_the_methods_is_the_one_compared_to(
    observations: pl.DataFrame,
) -> None:
    assert reference_among_methods(observations, ConfiguredReference(AMG)) == AMG


def test_a_configured_reference_with_no_system_in_the_group_gives_no_comparison(
    observations: pl.DataFrame,
) -> None:
    assert reference_among_methods(observations, ConfiguredReference("elsewhere")) is None


def test_empty_group_yields_no_comparison() -> None:
    assert reference_among_methods(synthetic_observations([]), NoReferenceConfigured()) is None
