"""Tests for solve-time/memory measurement in ``run_cg_comparison``."""

from __future__ import annotations

import torch

from neuralls.domain.solver.comparison import run_cg_comparison


def test_solve_measures_time_and_memory(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    """Every successful solve records a non-negative solve time and peak memory,
    and never touches the setup fields (measured by the composition layer, not here).
    """
    results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners)

    for result in results.values():
        assert result.solve_time_seconds is not None
        assert result.solve_time_seconds >= 0
        assert result.solve_peak_memory_bytes is not None
        assert result.solve_peak_memory_bytes >= 0
        assert result.setup_cost is None


def test_avg_iteration_time_derives_from_solve_time_and_iterations(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    """The averaged per-iteration cost is solve_time / iterations, not a stored field."""
    results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners)

    for result in results.values():
        assert result.iterations > 0
        assert result.solve_time_seconds is not None
        expected = result.solve_time_seconds / result.iterations
        assert result.avg_iteration_time_seconds == expected


def test_total_time_is_none_without_setup_measurement(
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    """total_time_seconds falls back to solve time alone when setup was never measured."""
    results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners)

    for result in results.values():
        assert result.total_time_seconds == result.solve_time_seconds
