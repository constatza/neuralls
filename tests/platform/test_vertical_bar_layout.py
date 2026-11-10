"""Sizing and annotation placement of the vertical bar figures, computed as data."""

from __future__ import annotations

import pytest

from neuralls.platform.reporting.figure_annotations import ColumnAnnotation, value_axis_top
from neuralls.platform.reporting.figure_sizing import bar_columns_figure_size


def test_the_width_grows_with_the_number_of_methods_past_a_minimum() -> None:
    narrow, _ = bar_columns_figure_size(["a"], 30.0)
    few, _ = bar_columns_figure_size(["a"] * 6, 30.0)
    many, _ = bar_columns_figure_size(["a"] * 12, 30.0)

    assert narrow < few < many


def test_the_width_does_not_depend_on_label_length() -> None:
    short, _ = bar_columns_figure_size(["a"] * 8, 30.0)
    long, _ = bar_columns_figure_size(["a long method label"] * 8, 30.0)

    assert short == long


def test_the_height_grows_with_the_longest_label_line_when_rotated() -> None:
    _, short = bar_columns_figure_size(["amg"], 45.0)
    _, long = bar_columns_figure_size(["amg with a long suffix"], 45.0)

    assert short < long


def test_unrotated_labels_need_height_for_their_stacked_lines_only() -> None:
    _, one_line = bar_columns_figure_size(["a long single line"], 0.0)
    _, same_one_line = bar_columns_figure_size(["a"], 0.0)
    _, two_lines = bar_columns_figure_size(["a\nb"], 0.0)

    assert one_line == same_one_line
    assert one_line < two_lines


def test_annotation_text_is_wrapped_to_about_a_column() -> None:
    annotation = ColumnAnnotation(0, 1.0, "3/4 returned an iterate")

    assert annotation.wrapped == "3/4 returned an\niterate"


def test_the_value_axis_leaves_room_above_the_tallest_annotation_base() -> None:
    annotations = [ColumnAnnotation(0, 2.0, "4/9 solved"), ColumnAnnotation(1, 5.0, "9/9 solved")]

    top = value_axis_top(annotations)

    assert top is not None
    assert top > 5.0


def test_more_annotation_lines_need_more_headroom() -> None:
    one_line = value_axis_top([ColumnAnnotation(0, 5.0, "4/9 solved")])
    three_lines = value_axis_top(
        [ColumnAnnotation(0, 5.0, "total unknown: generation unavailable")]
    )

    assert one_line is not None
    assert three_lines is not None
    assert one_line < three_lines


@pytest.mark.parametrize("bases", [[], [0.0, 0.0]])
def test_there_is_no_scale_to_extend_when_no_mark_rises_above_zero(bases: list[float]) -> None:
    annotations = [ColumnAnnotation(i, base, "0/9 solved") for i, base in enumerate(bases)]

    assert value_axis_top(annotations) is None
