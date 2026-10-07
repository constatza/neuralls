"""Tests for ``CGComparisonResult``'s derived cost properties."""

from __future__ import annotations

from neuralls.domain.solver.models.result import CGComparisonResult


def test_total_time_sums_setup_and_solve(
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    assert comparison_result_with_cost_data.total_time_seconds == 2.0 + 1.0


def test_total_time_is_zero_when_nothing_else_measured(
    comparison_result_without_cost_data: CGComparisonResult,
) -> None:
    """``setup_cost`` is always measured (0.0 default), so ``total_time_seconds``
    is never ``None`` — it is 0.0 when generation/solve were never measured."""
    assert comparison_result_without_cost_data.total_time_seconds == 0.0


def test_avg_iteration_time_divides_solve_time_by_iterations(
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    assert comparison_result_with_cost_data.avg_iteration_time_seconds == 1.0 / 5


def test_avg_iteration_time_none_with_zero_iterations(
    comparison_result_with_zero_iterations: CGComparisonResult,
) -> None:
    assert comparison_result_with_zero_iterations.avg_iteration_time_seconds is None


def test_peak_memory_bytes_takes_max_of_setup_and_solve(
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    assert comparison_result_with_cost_data.peak_memory_bytes == 2000


def test_peak_memory_bytes_none_when_neither_measured(
    comparison_result_without_cost_data: CGComparisonResult,
) -> None:
    assert comparison_result_without_cost_data.peak_memory_bytes is None
