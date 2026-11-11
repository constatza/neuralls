"""Sizing and annotation placement of the vertical bar figures, computed as data."""

from __future__ import annotations

import pytest

from neuralls.platform.reporting.figure_annotations import ColumnAnnotation, value_axis_top
from neuralls.platform.reporting.figure_sizing import (
    AXIS_LABEL_WRAP,
    bar_columns_figure_size,
    paired_scatter_figure_size,
    rotated_label_overhang_in,
    with_outside_legend,
    wrap_axis_label,
)


def test_the_width_grows_with_the_number_of_methods_past_a_minimum() -> None:
    narrow, _ = bar_columns_figure_size(["a"], 30.0)
    few, _ = bar_columns_figure_size(["a"] * 6, 30.0)
    many, _ = bar_columns_figure_size(["a"] * 12, 30.0)

    assert narrow < few < many


def test_short_labels_do_not_widen_the_figure() -> None:
    plain, _ = bar_columns_figure_size(["a"] * 8, 30.0)
    short, _ = bar_columns_figure_size(["amg"] * 8, 30.0)

    assert plain == short


def test_a_label_overhanging_the_first_column_widens_the_figure() -> None:
    plain, _ = bar_columns_figure_size(["a", "b"], 30.0)
    wide, _ = bar_columns_figure_size(["a $k=20$ with a long suffix of text", "b"], 30.0)

    assert wide > plain


def test_a_label_that_stays_under_its_own_column_has_no_overhang() -> None:
    assert rotated_label_overhang_in(["a", "b", "c"], 30.0) == 0.0


def test_the_overhang_shrinks_for_labels_further_right() -> None:
    first = rotated_label_overhang_in(["x" * 22, "a", "a"], 30.0)
    last = rotated_label_overhang_in(["a", "a", "x" * 22], 30.0)

    assert first > last


def test_plain_labels_are_wrapped_at_spaces_and_mathtext_lines_are_kept() -> None:
    plain = wrap_axis_label("Incomplete Cholesky with level of fill")
    math = wrap_axis_label("Neural AMG preconditioner ($k=20$, $\\mu=0$)")

    assert plain == "Incomplete Cholesky with\nlevel of fill"
    assert max(len(line) for line in plain.split("\n")) <= AXIS_LABEL_WRAP
    assert "\n" not in math


def test_the_paired_scatter_grows_by_its_legend_only_sideways() -> None:
    narrow = paired_scatter_figure_size(["a"])
    wide = paired_scatter_figure_size(["a long method label"])

    assert wide[0] > narrow[0]
    assert wide[1] == narrow[1]


def test_an_outside_legend_widens_a_figure_and_keeps_its_height() -> None:
    assert with_outside_legend((8.0, 5.0), ["a\nlonger line"])[1] == 5.0
    assert with_outside_legend((8.0, 5.0), ["a\nlonger line"])[0] > 8.0
    assert with_outside_legend((8.0, 5.0), []) == (8.0, 5.0)


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

    assert annotation.wrapped == "3/4\nreturned\nan\niterate"


def test_annotation_lines_are_wrapped_separately_and_kept_apart() -> None:
    annotation = ColumnAnnotation(0, 1.0, "102.5 s\nfit inline, counted in setup")

    assert annotation.wrapped == "102.5 s\nfit\ninline,\ncounted\nin setup"


def test_alternatives_around_a_slash_are_not_split() -> None:
    annotation = ColumnAnnotation(0, 1.0, "total unknown: offline fit / training unavailable")

    assert annotation.wrapped == "total\nunknown:\noffline\nfit / training\nunavailable"


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


def test_headroom_follows_the_lines_a_base_needs_not_the_tallest_base() -> None:
    short_bar_long_note = ColumnAnnotation(
        0, 1.0, "total unknown: offline fit / training unavailable"
    )
    tall_bar_short_note = ColumnAnnotation(1, 10.0, "9/9 solved")

    top = value_axis_top([short_bar_long_note, tall_bar_short_note])
    alone = value_axis_top([tall_bar_short_note])

    assert top == alone
