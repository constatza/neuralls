"""Tests for the work-precision (final precision vs. total wall time) plot."""

from __future__ import annotations

from pathlib import Path

from neuralls.domain.solver.models.result import CGComparisonResult
from neuralls.platform.reporting.plots import plot_work_precision


def test_work_precision_plot_is_written(
    tmp_path: Path,
    comparison_result_with_cost_data: CGComparisonResult,
) -> None:
    target = tmp_path / "work_precision.png"
    plot_work_precision({"jacobi": comparison_result_with_cost_data}, save_path=target)
    assert target.stat().st_size > 0


def test_work_precision_plot_skips_entries_without_cost_data(
    tmp_path: Path,
    comparison_result_with_cost_data: CGComparisonResult,
    comparison_result_without_cost_data: CGComparisonResult,
) -> None:
    """A method with no measured cost is dropped, not plotted at a fake x=0."""
    target = tmp_path / "work_precision_mixed.png"
    plot_work_precision(
        {"jacobi": comparison_result_with_cost_data, "none": comparison_result_without_cost_data},
        save_path=target,
    )
    assert target.stat().st_size > 0
