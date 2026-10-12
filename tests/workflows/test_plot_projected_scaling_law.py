"""Tests for `plot_projected_scaling_law`."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from neuralls.domain.solver.models.result import ProjectedComparisonResult, SetupSharingPolicy
from neuralls.platform.reporting.plots import plot_projected_scaling_law


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
