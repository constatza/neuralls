"""Tests for `plot_projected_scaling_law`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from neuralls.domain.solver.models.result import ProjectedComparisonResult, SetupSharingPolicy
from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.platform.reporting.plots import (
    plot_convergence_comparison,
    plot_projected_scaling_law,
    plot_system_time_scatter,
)


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


@pytest.fixture
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """Keep figures alive past the plotting call so tests can inspect their artists."""
    figures: list[Figure] = []

    def keep_open(fig: Figure) -> None:
        figures.append(fig)

    monkeypatch.setattr(plt, "close", keep_open)
    return figures


def _legend_texts(fig: Figure) -> list[str]:
    legend = fig.axes[0].get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def test_scaling_law_legend_has_labels_n_and_std_band_entry(
    captured_figures: list[Figure],
) -> None:
    with_std = replace(
        _projected("neural", share_setup=True, share_generation=True),
        solve_time_std_seconds=0.4,
    )
    plot_projected_scaling_law(
        {"neural": with_std},
        labels={"neural": "Neural (CG)"},
        subtitle="fine level, N = 5",
    )

    (fig,) = captured_figures
    assert _legend_texts(fig) == ["Neural (CG) (n=5)", "±1σ over systems"]
    assert fig.axes[0].get_title() == "fine level, N = 5"


def test_scaling_law_without_std_has_no_band_entry(captured_figures: list[Figure]) -> None:
    single = replace(_projected("amg", share_setup=False, share_generation=False), n_systems=1)
    plot_projected_scaling_law({"amg": single})

    (fig,) = captured_figures
    assert _legend_texts(fig) == ["amg (n=1)"]


def test_scaling_law_style_matches_convergence_plot(captured_figures: list[Figure]) -> None:
    families = {"jacobi": PreconditionerType.JACOBI, "amg": PreconditionerType.AMG}
    plot_convergence_comparison(
        {name: {"residual_history_rel": [1.0, 0.1]} for name in families},
        families=families,
    )
    plot_projected_scaling_law(
        {name: _projected(name, share_setup=True, share_generation=True) for name in families},
        families=families,
    )

    convergence_fig, scaling_fig = captured_figures
    expected = {line.get_label(): line for line in convergence_fig.axes[0].get_lines()}
    for line in scaling_fig.axes[0].get_lines():
        base = expected[str(line.get_label()).split(" (n=")[0]]
        assert line.get_color() == base.get_color()
        assert line.get_marker() == base.get_marker()


def test_scatter_uses_system_labels_and_labelled_legend(
    captured_figures: list[Figure],
) -> None:
    times = {"jacobi": {"s0": 1.0, "s1": 2.0}, "amg": {"s0": 0.5}}
    plot_system_time_scatter(
        times,
        labels={"amg": "AMG (POD-2G)"},
        system_labels={"s0": "A0/r0"},
        subtitle="coarse level",
    )

    (fig,) = captured_figures
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A0/r0", "s1"]
    assert _legend_texts(fig) == ["AMG (POD-2G) (n=1)", "jacobi (n=2)"]
    assert ax.get_title() == "coarse level"
