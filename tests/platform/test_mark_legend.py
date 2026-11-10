"""The legend names exactly the marks a figure drew, in a fixed order and with no counts."""

from __future__ import annotations

from collections.abc import Collection

from neuralls.platform.reporting.mark_legend import Mark, mark_handles


def _labels(marks: Collection[Mark]) -> list[str]:
    return [str(handle.get_label()) for handle in mark_handles(marks)]


def test_a_box_over_points_is_explained_by_its_four_marks() -> None:
    marks = {Mark.SYSTEM_POINTS, Mark.QUANTILE_WHISKER, Mark.MEDIAN, Mark.IQR}

    assert _labels(marks) == ["IQR (q25-q75)", "median", "whiskers within q05-q95", "systems"]


def test_a_mean_bar_without_an_interval_has_no_interval_entry() -> None:
    assert _labels({Mark.MEAN_BAR}) == ["mean"]


def test_a_mean_bar_with_an_interval_and_points() -> None:
    marks = {Mark.SYSTEM_POINTS, Mark.CONFIDENCE_INTERVAL, Mark.MEAN_BAR}

    assert _labels(marks) == ["mean", "95% matrix-cluster interval", "systems"]


def test_a_ratio_bar_is_not_called_a_mean() -> None:
    assert _labels({Mark.RATIO_BAR}) == ["ratio of sums"]


def test_a_convergence_factor_figure_names_its_mean_dot_and_median_diamond() -> None:
    marks = {Mark.MEDIAN_POINT, Mark.MEAN_POINT, Mark.CONFIDENCE_INTERVAL, Mark.SYSTEM_POINTS}

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
        "IQR (q25-q75)",
        "median",
        "median",
        "density of systems",
        "whiskers within q05-q95",
        "mean",
        "systems",
        "matrix means",
    ]
