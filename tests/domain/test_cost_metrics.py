"""Tests for ``neuralls.domain.solver.cost_metrics``' pure size-normalized/throughput functions."""

from __future__ import annotations

import pytest

from neuralls.domain.solver.cost_metrics import (
    generation_time_per_dof,
    iterations_per_second,
    peak_memory_per_dof,
    setup_time_per_dof,
    time_per_dof_per_iteration,
)
from neuralls.domain.solver.models.result import CGComparisonResult


def test_iterations_per_second_computes_throughput(
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    assert iterations_per_second(comparison_result_with_cost_data) == 5 / 1.0


def test_iterations_per_second_none_without_solve_time(
    comparison_result_without_cost_data: CGComparisonResult,
) -> None:
    assert iterations_per_second(comparison_result_without_cost_data) is None


def test_iterations_per_second_none_with_zero_iterations(
    comparison_result_with_zero_iterations: CGComparisonResult,
) -> None:
    assert iterations_per_second(comparison_result_with_zero_iterations) is None


def test_time_per_dof_per_iteration_normalizes_by_system_size(
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    # avg_iteration_time_seconds = 1.0 / 5 = 0.2
    assert (
        time_per_dof_per_iteration(comparison_result_with_cost_data, system_size=100) == 0.2 / 100
    )


def test_time_per_dof_per_iteration_none_without_solve_time(
    comparison_result_without_cost_data: CGComparisonResult,
) -> None:
    assert time_per_dof_per_iteration(comparison_result_without_cost_data, system_size=100) is None


def test_time_per_dof_per_iteration_rejects_nonpositive_system_size(
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    with pytest.raises(ValueError, match="system_size must be positive"):
        time_per_dof_per_iteration(comparison_result_with_cost_data, system_size=0)


def test_setup_time_per_dof_normalizes_by_system_size(
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    assert setup_time_per_dof(comparison_result_with_cost_data, system_size=100) == 2.0 / 100


def test_setup_time_per_dof_zero_without_setup_time(
    comparison_result_without_cost_data: CGComparisonResult,
) -> None:
    """`setup_cost` defaults to 0.0 (always measured), never `None`."""
    assert setup_time_per_dof(comparison_result_without_cost_data, system_size=100) == 0.0


def test_peak_memory_per_dof_normalizes_by_system_size(
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    # peak_memory_bytes = max(1000, 2000) = 2000
    assert peak_memory_per_dof(comparison_result_with_cost_data, system_size=100) == 2000 / 100


def test_peak_memory_per_dof_none_without_memory_data(
    comparison_result_without_cost_data: CGComparisonResult,
) -> None:
    assert peak_memory_per_dof(comparison_result_without_cost_data, system_size=100) is None


def test_generation_time_per_dof_normalizes_by_system_size(
    comparison_result_with_generation_cost: CGComparisonResult,
) -> None:
    assert (
        generation_time_per_dof(comparison_result_with_generation_cost, system_size=100)
        == 5.0 / 100
    )


def test_generation_time_per_dof_none_without_generation_cost(
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    assert generation_time_per_dof(comparison_result_with_cost_data, system_size=100) is None
