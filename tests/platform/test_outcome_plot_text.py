"""Legend wording and curve geometry of the group outcome plots, as pure functions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from neuralls.domain.benchmarking.profiles import ProfileSeries
from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.platform.reporting import profile_plots
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.platform.reporting.paired_plots import (
    FAILED_AXIS_MARGIN,
    PairedLogAxes,
    paired_log_axes,
    speed_ratio_legend,
)
from neuralls.platform.reporting.profile_plots import (
    excluded_zero_best_note,
    profile_step_curve,
)
from neuralls.platform.reporting.workload_bars import (
    BarSegment,
    WorkloadTotal,
    workload_bar_label,
    workload_segments,
)

M = FAILED_AXIS_MARGIN


@pytest.fixture
def profile_series() -> ProfileSeries:
    return ProfileSeries(pl.DataFrame(), pl.DataFrame(), 0)


@pytest.fixture
def profile_style(group_style_of: Callable[..., GroupStyle]) -> GroupStyle:
    return group_style_of("amg")


@pytest.fixture
def profile_target() -> FigureTarget:
    return FigureTarget(Path("unused"), "profile")


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (MetricId.SOLVE_S, r"Solve-only time relative to the best method, $\tau$"),
        (MetricId.ONLINE_S, r"Amortized online time relative to the best method, $\tau$"),
    ],
)
def test_performance_profile_axis_names_its_timing_metric(
    metric: MetricId,
    expected: str,
    profile_series: ProfileSeries,
    profile_style: GroupStyle,
    profile_target: FigureTarget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []

    def record(*args: object, xlabel: str, **kwargs: object) -> FigureTarget:
        labels.append(xlabel)
        return profile_target

    monkeypatch.setattr(profile_plots, "_step_profile_figure", record)
    assert (
        profile_plots.plot_performance_profile(
            profile_series, metric=metric, style=profile_style, target=profile_target
        )
        == profile_target
    )
    assert labels == [expected]


@pytest.mark.parametrize(
    ("median", "ci", "expected"),
    [
        (1.234, (0.98, 1.5), "AMG: median 1.23 [0.98, 1.50]"),
        (2.0, None, "AMG: median 2.00"),
        (None, None, "AMG: no paired ratio"),
    ],
)
def test_speed_ratio_legend_states_the_median_and_interval_it_has(
    median: float | None, ci: tuple[float, float] | None, expected: str
) -> None:
    assert speed_ratio_legend("AMG", median=median, ci=ci) == expected


@pytest.mark.parametrize(
    ("n_excluded", "expected"),
    [
        (0, None),
        (1, "1 system excluded: best value is 0"),
        (3, "3 systems excluded: best value is 0"),
    ],
)
def test_excluded_zero_best_note_appears_only_when_systems_were_excluded(
    n_excluded: int, expected: str | None
) -> None:
    assert excluded_zero_best_note(n_excluded) == expected


def test_profile_step_curve_rises_from_zero_and_extends_its_plateau_to_the_shared_end() -> None:
    xs, ys = profile_step_curve(
        np.array([2.0, 3.0]), np.array([1 / 4, 2 / 4]), x_min=1.0, x_max=10.0
    )

    assert xs.tolist() == [2.0, 2.0, 3.0, 10.0]
    assert ys.tolist() == [0.0, 0.25, 0.5, 0.5]


def test_profile_step_curve_of_a_method_that_solved_nothing_is_flat_zero() -> None:
    xs, ys = profile_step_curve(np.array([]), np.array([]), x_min=1.0, x_max=10.0)

    assert xs.tolist() == [1.0, 10.0]
    assert ys.tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("total", "segments"),
    [
        (WorkloadTotal.SOLVE, (BarSegment.CONVERGED, BarSegment.FAILED)),
        (WorkloadTotal.ONLINE, (BarSegment.SETUP, BarSegment.CONVERGED, BarSegment.FAILED)),
    ],
)
def test_a_workload_bar_stacks_the_table_columns_that_sum_to_its_total(
    total: WorkloadTotal, segments: tuple[BarSegment, ...]
) -> None:
    assert workload_segments(total) == segments


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((3, 4, 4), "3/4 converged"),
        ((1, 2, 4), "1/4 converged, 2/4 attempted"),
        ((0, 0, 4), "0/4 converged, 0/4 attempted"),
    ],
)
def test_a_workload_bar_label_counts_converged_and_short_attempts_of_the_planned_rhs(
    counts: tuple[int, int, int], expected: str
) -> None:
    n_converged, n_attempted, n_planned = counts

    assert (
        workload_bar_label(n_converged=n_converged, n_attempted=n_attempted, n_planned=n_planned)
        == expected
    )


def test_paired_log_axes_span_the_positive_times_and_put_failures_on_the_upper_edge() -> None:
    axes = paired_log_axes([0.5, 2.0, 8.0])

    assert axes == PairedLogAxes(low=0.5 / M, high=8.0 * M, edge=8.0 * M)


def test_paired_log_axes_ignore_times_a_log_axis_cannot_place() -> None:
    assert paired_log_axes([0.0, 2.0]) == PairedLogAxes(low=2.0 / M, high=2.0 * M, edge=2.0 * M)


def test_paired_log_axes_need_a_positive_time() -> None:
    assert paired_log_axes([0.0]) is None
