"""Tests for ``neuralls.domain.solver.cost_metrics``' pure size-normalized/throughput functions."""

from __future__ import annotations

from neuralls.domain.solver.cost_metrics import (
    generation_time_per_dof,
    iterations_per_second,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult


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
