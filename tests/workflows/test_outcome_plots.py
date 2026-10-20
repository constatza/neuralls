"""Group outcome plots: they draw (including failures) and never impute a time.

The tests read the plotted data back from the captured matplotlib figure, so a
plot that saves a file but draws the wrong points fails.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuralls.domain.solver.models.outcomes import Failed, Solved, SystemOutcome
from neuralls.domain.solver.models.statistics import Interval, PairedComparison
from neuralls.platform.reporting import outcome_plots
from neuralls.platform.reporting.outcome_plots import (
    GroupStyle,
    plot_ecdf_profile,
    plot_paired_time_scatter,
    plot_performance_profile,
    plot_speedup_cdf,
    plot_time_scatter,
)
from neuralls.platform.reporting.plots import finish_group_figure
from neuralls.shared.types import MatrixId

type Outcomes = dict[str, dict[str, SystemOutcome]]


def _solved(iterations: int, seconds: float) -> Solved:
    return Solved(MatrixId(7), iterations, solve_s=seconds, amortized_online_s=seconds)


def _failed() -> Failed:
    return Failed(MatrixId(7), iterations=100, reason="iteration limit")


@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """Every figure the outcome plots finish, kept so tests can read the drawn data."""
    captured: list[Figure] = []

    def capture(fig: Figure, *args, **kwargs) -> None:
        captured.append(fig)
        finish_group_figure(fig, *args, **kwargs)

    monkeypatch.setattr(outcome_plots, "finish_group_figure", capture)
    return captured


@pytest.fixture
def outcomes() -> Outcomes:
    return {
        "amg": {"s0": _solved(5, 1.0), "s1": _solved(6, 2.0), "s2": _solved(7, 4.0)},
        "jacobi": {"s0": _solved(50, 2.0), "s1": _failed(), "s2": _failed()},
    }


@pytest.fixture
def zero_cost_reference_outcomes() -> Outcomes:
    """A zero-time reference observation is not a comparable positive ratio."""
    return {
        "amg": {"s0": _solved(5, 1.0), "s1": _solved(6, 0.0), "s2": _solved(7, 4.0)},
        "jacobi": {"s0": _solved(50, 2.0), "s1": _solved(9, 3.0), "s2": _failed()},
    }


@pytest.fixture
def style() -> GroupStyle:
    return GroupStyle(title="group", labels={"amg": "AMG"})


def _axes(figures: list[Figure]) -> Axes:
    return figures[-1].axes[0]


def _hollow_offsets(ax: Axes) -> np.ndarray:
    """Coordinates of every unfilled scatter marker."""
    hollow = [c.get_offsets() for c in ax.collections if len(c.get_facecolor()) == 0]
    return np.vstack(hollow) if hollow else np.empty((0, 2))


def test_every_outcome_plot_returns_its_saved_file(
    tmp_path: Path, outcomes: Outcomes, style: GroupStyle
) -> None:
    paths = [
        plot_performance_profile(outcomes, style=style, save_path=tmp_path / "perf.png"),
        plot_ecdf_profile(outcomes, style=style, save_path=tmp_path / "ecdf.png"),
        plot_time_scatter(outcomes, style=style, save_path=tmp_path / "scatter.png"),
        plot_paired_time_scatter(outcomes, style=style, save_path=tmp_path / "paired.png"),
    ]

    assert [p.name for p in paths if p is not None] == [
        "perf.png", "ecdf.png", "scatter.png", "paired.png"
    ]  # fmt: skip
    assert all(p.exists() for p in paths if p is not None)


def test_ecdf_steps_end_at_the_success_rate_and_the_shared_x_max(
    tmp_path: Path, outcomes: Outcomes, style: GroupStyle, figures: list[Figure]
) -> None:
    plot_ecdf_profile(outcomes, style=style, save_path=tmp_path / "e.png")

    amg, jacobi = _axes(figures).lines
    assert list(amg.get_xdata()) == [5, 5, 6, 7, 50]
    assert list(amg.get_ydata()) == pytest.approx([0, 1 / 3, 2 / 3, 1, 1])
    assert list(jacobi.get_xdata()) == [50, 50, 50]
    assert list(jacobi.get_ydata()) == pytest.approx([0, 1 / 3, 1 / 3])


def test_performance_profile_plateaus_at_the_success_rate(
    tmp_path: Path, outcomes: Outcomes, style: GroupStyle, figures: list[Figure]
) -> None:
    plot_performance_profile(outcomes, style=style, save_path=tmp_path / "p.png")

    amg, jacobi = _axes(figures).lines
    assert np.asarray(amg.get_ydata())[-1] == pytest.approx(1.0)
    assert np.asarray(jacobi.get_ydata())[-1] == pytest.approx(1 / 3)
    assert np.asarray(amg.get_xdata())[-1] == np.asarray(jacobi.get_xdata())[-1]


def test_profile_of_a_method_that_solved_nothing_is_a_flat_zero_line(
    tmp_path: Path, style: GroupStyle, figures: list[Figure]
) -> None:
    all_failed: Outcomes = {"jacobi": {"s0": _failed(), "s1": _failed()}}

    path = plot_ecdf_profile(all_failed, style=style, save_path=tmp_path / "e.png")

    (line,) = _axes(figures).lines
    assert path is not None and list(line.get_ydata()) == [0.0, 0.0]


def test_time_scatter_puts_failures_at_the_top_edge(
    tmp_path: Path, outcomes: Outcomes, style: GroupStyle, figures: list[Figure]
) -> None:
    plot_time_scatter(outcomes, style=style, save_path=tmp_path / "s.png")

    edge = 4.0 * outcome_plots._FAILED_AXIS_MARGIN
    assert _hollow_offsets(_axes(figures)).tolist() == [[1, edge], [2, edge]]


def test_time_scatter_handles_a_group_where_nothing_was_solved(
    tmp_path: Path, style: GroupStyle
) -> None:
    path = plot_time_scatter(
        {"jacobi": {"s0": _failed(), "s1": _failed()}},
        style=style,
        save_path=tmp_path / "s.png",
    )

    assert path is not None and path.exists()


@pytest.mark.real_render
def test_time_scatter_saves_a_decodable_png(
    tmp_path: Path, style: GroupStyle, outcomes: Outcomes
) -> None:
    path = plot_time_scatter(outcomes, style=style, save_path=tmp_path / "s.png")

    assert path is not None
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_paired_scatter_marks_the_reference_runs_the_method_failed(
    tmp_path: Path, outcomes: Outcomes, style: GroupStyle, figures: list[Figure]
) -> None:
    plot_paired_time_scatter(outcomes, style=style, save_path=tmp_path / "p.png")

    ax = _axes(figures)
    edge = 4.0 * outcome_plots._FAILED_AXIS_MARGIN
    assert _hollow_offsets(ax).tolist() == [[2.0, edge], [4.0, edge]]
    filled = [
        np.asarray(c.get_offsets()).tolist() for c in ax.collections if len(c.get_facecolor())
    ]
    assert filled == [[[1.0, 2.0]]]


def test_paired_scatter_does_not_mark_a_zero_cost_reference_run_as_failed(
    tmp_path: Path,
    zero_cost_reference_outcomes: Outcomes,
    style: GroupStyle,
    figures: list[Figure],
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot_paired_time_scatter(
            zero_cost_reference_outcomes, style=style, save_path=tmp_path / "p.png"
        )

    ax = _axes(figures)
    edge = 4.0 * outcome_plots._FAILED_AXIS_MARGIN
    assert _hollow_offsets(ax).tolist() == [[4.0, edge]]
    filled = [
        np.asarray(c.get_offsets()).tolist() for c in ax.collections if len(c.get_facecolor())
    ]
    assert filled == [[[1.0, 2.0]]]
    assert ax.get_xlim()[0] > 0
    assert ax.get_ylim()[0] > 0
    assert not any("non-positive" in str(item.message) for item in caught)


def test_paired_scatter_returns_none_when_all_solved_times_are_zero(
    tmp_path: Path, style: GroupStyle
) -> None:
    outcomes: Outcomes = {
        "amg": {"s0": _solved(5, 0.0)},
        "jacobi": {"s0": _solved(6, 0.0)},
    }

    path = plot_paired_time_scatter(outcomes, style=style, save_path=tmp_path / "p.png")

    assert path is None and not (tmp_path / "p.png").exists()


def test_paired_scatter_draws_nothing_for_a_single_method(
    tmp_path: Path, style: GroupStyle
) -> None:
    path = plot_paired_time_scatter(
        {"amg": {"s0": _solved(5, 1.0)}}, style=style, save_path=tmp_path / "p.png"
    )

    assert path is None and not (tmp_path / "p.png").exists()


@pytest.fixture
def comparisons() -> dict[str, PairedComparison]:
    return {
        "jacobi": PairedComparison(
            speed_ratios=(0.5, 2.0, 4.0),
            median_speed_ratio=2.0,
            median_speed_ratio_ci=Interval(0.5, 4.0, 0.95),
            n_both_solved=3,
            n_reference_only=0,
            n_method_only=0,
            wilcoxon_p=None,
        )
    }


def test_speedup_cdf_has_the_sorted_speedups_and_a_line_at_one(
    tmp_path: Path, comparisons, style: GroupStyle, figures: list[Figure]
) -> None:
    path = plot_speedup_cdf(comparisons, style=style, save_path=tmp_path / "c.png")

    ax = _axes(figures)
    curve, unity = ax.lines
    assert path is not None
    assert list(curve.get_xdata()) == [0.5, 2.0, 4.0]
    assert list(curve.get_ydata()) == pytest.approx([1 / 3, 2 / 3, 1])
    assert list(unity.get_xdata()) == [1.0, 1.0]
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_texts()[0].get_text() == "jacobi: median 2.00 [0.50, 4.00], n=3"


def test_speedup_cdf_draws_nothing_without_comparisons(tmp_path: Path, style: GroupStyle) -> None:
    assert plot_speedup_cdf({}, style=style, save_path=tmp_path / "c.png") is None
