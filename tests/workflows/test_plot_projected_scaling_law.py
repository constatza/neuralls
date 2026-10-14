"""Tests for `plot_projected_scaling_law`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from neuralls.domain.solver.models.result import ProjectedComparisonResult, SetupSharingPolicy
from neuralls.platform.reporting.plots import plot_projected_scaling_law, plot_system_time_scatter


def _projected(
    name: str, *, share_setup: bool, share_generation: bool
) -> ProjectedComparisonResult:
    return ProjectedComparisonResult(
        preconditioner_name=name,
        n_systems=5,
        setup_sharing=SetupSharingPolicy(
            share_setup=share_setup, share_generation=share_generation
        ),
        setup_cost_total_seconds=10.0,
        generation_cost_total_seconds=50.0,
        total_solve_time_seconds=5.0,
        total_iterations=50,
    )


def test_plot_projected_scaling_law_saves_file(tmp_path: Path) -> None:
    save_path = tmp_path / "scaling.png"
    projected = {
        "neural": _projected("neural", share_setup=True, share_generation=True),
        "amg": _projected("amg", share_setup=False, share_generation=False),
    }

    plot_projected_scaling_law(projected, save_path=save_path)

    assert save_path.exists()


def test_plot_projected_scaling_law_handles_empty_mapping(tmp_path: Path) -> None:
    """An empty projection mapping must not crash (e.g. no legend to draw)."""
    save_path = tmp_path / "scaling.png"

    plot_projected_scaling_law({}, save_path=save_path)

    assert save_path.exists()


def test_plot_projected_scaling_law_draws_a_std_band_when_present(tmp_path: Path) -> None:
    with_std = replace(
        _projected("neural", share_setup=True, share_generation=True),
        solve_time_std_seconds=0.4,
    )
    save_path = tmp_path / "scaling_band.png"

    plot_projected_scaling_law({"neural": with_std}, save_path=save_path)

    assert save_path.exists()


def test_plot_system_time_scatter_saves_file_with_unequal_system_sets(tmp_path: Path) -> None:
    """A preconditioner missing some systems (failed/incomparable) still plots."""
    save_path = tmp_path / "scatter.png"
    times = {"jacobi": {"s0": 1.0, "s1": 2.0, "s2": 1.5}, "amg": {"s0": 0.5, "s2": 0.7}}

    plot_system_time_scatter(times, save_path=save_path)

    assert save_path.exists()


def test_plot_system_time_scatter_handles_empty_mapping(tmp_path: Path) -> None:
    save_path = tmp_path / "scatter_empty.png"

    plot_system_time_scatter({}, save_path=save_path)

    assert save_path.exists()
