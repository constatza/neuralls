"""Tests for the energy-error convergence plot."""

from __future__ import annotations

from pathlib import Path

import torch

from neuralls.domain.solver.comparison import run_cg_comparison
from neuralls.platform.reporting.plots import plot_error_convergence_comparison


def test_error_plot_is_written(
    tmp_path: Path,
    spd_matrix: torch.Tensor,
    rhs: torch.Tensor,
    preconditioners: dict[str, object],
) -> None:
    results = run_cg_comparison(spd_matrix, rhs, preconditioners=preconditioners)
    target = tmp_path / "error.png"
    plot_error_convergence_comparison(results, save_path=target)
    assert target.stat().st_size > 0
