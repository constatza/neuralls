"""The legend names exactly the marks a figure drew, in a fixed order and with no counts."""

from __future__ import annotations

from collections.abc import Collection

import pytest

from neuralls.domain.benchmarking.uncertainty import BootstrapKind
from neuralls.platform.reporting.mark_legend import Mark, interval_mark, mark_handles


def _labels(marks: Collection[Mark]) -> list[str]:
    return [str(handle.get_label()) for handle in mark_handles(marks)]


def test_a_box_over_points_is_explained_by_its_four_marks() -> None:
    marks = {Mark.SYSTEM_POINTS, Mark.QUANTILE_WHISKER, Mark.MEDIAN, Mark.IQR}

    assert _labels(marks) == ["IQR (q25-q75)", "median", "whiskers within q05-q95", "systems"]


def test_a_mean_bar_without_an_interval_has_no_interval_entry() -> None:
    assert _labels({Mark.MEAN_BAR}) == ["mean"]


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        (BootstrapKind.MATRIX_CLUSTER, "95% matrix-cluster interval"),
        (BootstrapKind.TWO_LEVEL, "95% two-level interval"),
    ],
)
def test_a_mean_bar_with_an_interval_and_points_names_its_bootstrap(
    kind: BootstrapKind, text: str
) -> None:
    marks = {Mark.SYSTEM_POINTS, interval_mark(kind), Mark.MEAN_BAR}

    assert _labels(marks) == ["mean", text, "systems"]


def test_a_ratio_bar_is_not_called_a_mean() -> None:
    assert _labels({Mark.RATIO_BAR}) == ["ratio of sums"]


def test_a_convergence_factor_figure_names_its_mean_dot_and_median_diamond() -> None:
    marks = {Mark.MEDIAN_POINT, Mark.MEAN_POINT, Mark.MATRIX_CLUSTER_INTERVAL, Mark.SYSTEM_POINTS}

    assert _labels(marks) == ["95% matrix-cluster interval", "median", "mean", "systems"]


def test_a_violin_column_explains_its_per_matrix_points() -> None:
    marks = {Mark.MATRIX_MEAN_POINTS, Mark.MEDIAN, Mark.VIOLIN}

    assert _labels(marks) == ["median", "density of systems", "matrix means"]


def test_no_marks_give_no_handles() -> None:
    assert mark_handles(set()) == []


def test_legend_text_states_no_sample_counts() -> None:
    assert [mark.text for mark in Mark] == [
        "mean",
        "ratio of sums",
        "95% matrix-cluster interval",
        "95% two-level interval",
        "IQR (q25-q75)",
        "median",
        "median",
        "density of systems",
        "whiskers within q05-q95",
        "mean",
        "systems",
        "matrix means",
    ]
