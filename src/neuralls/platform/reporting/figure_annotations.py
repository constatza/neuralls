"""The count text and per-row annotations that report figures write beside their marks."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

from matplotlib.axes import Axes

from neuralls.domain.solver.reporting.metrics import MetricId
from neuralls.platform.reporting.figure_text import LEGEND_FONT_SIZE

_ANNOTATION_ROOM = 1.45
"""Factor past the rightmost annotation start the x axis extends to, so "k/n" text fits inside."""


def counted(n: int, singular: str, plural: str) -> str:
    """``n`` with the noun in its singular or plural form."""
    return f"{n} {singular if n == 1 else plural}"


def count_label(n_defined: int, n_planned: int, metric: MetricId) -> str:
    """How many of ``n_planned`` units define ``metric``, in the metric's words: "k/n solved"."""
    return f"{n_defined}/{n_planned} {_defined_word(metric)}"


def _defined_word(metric: MetricId) -> str:
    match metric:
        case MetricId.ITERATIONS | MetricId.SOLVE_S | MetricId.ONLINE_S | MetricId.STANDALONE_S:
            return "solved"
        case MetricId.SETUP_S:
            return "set up"
        case MetricId.TIME_PER_ITERATION_S:
            return "returned an iterate"
        case MetricId.CONVERGENCE_FACTOR:
            return "converged with a residual history"
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class RowAnnotation:
    """The text right of a method's row, starting where the row's last mark ends."""

    row: int
    start: float
    text: str


def annotate_rows(ax: Axes, annotations: Sequence[RowAnnotation]) -> None:
    """Write each annotation and widen the x axis from 0 so the rightmost one fits inside."""
    for annotation in annotations:
        ax.text(
            annotation.start,
            annotation.row,
            f" {annotation.text}",
            va="center",
            fontsize=LEGEND_FONT_SIZE,
        )
    rightmost = max((annotation.start for annotation in annotations), default=0.0)
    if rightmost > 0.0:
        ax.set_xlim(0.0, rightmost * _ANNOTATION_ROOM)
    else:
        ax.set_xlim(left=0.0)
    ax.grid(True, axis="x", alpha=0.3)


_COLUMN_ANNOTATION_WRAP = 12
"""Characters per line of a text above a bar, narrow enough that neighbours' texts never touch."""

_NO_BREAK_SPACE = "\u00a0"

_COLUMN_LINE_FRACTION = 0.045
"""Fraction of the value axis' height one annotation line (and half a line of gap) takes."""

_MAX_ANNOTATION_FRACTION = 0.9
"""Most of the value axis an annotation may take, so a very tall note cannot flatten the bars."""

_COLUMN_ANNOTATION_GAP_PT = 2.0
"""Points between the top of a bar's last mark and its text."""


def _wrap_line(line: str) -> str:
    """Wrap ``line`` to a column, keeping the alternatives around a slash ("fit / training") whole.

    A word longer than the column stays on its own line instead of being cut.
    """
    glued = line.replace(" / ", f"{_NO_BREAK_SPACE}/{_NO_BREAK_SPACE}")
    wrapped = textwrap.fill(glued, _COLUMN_ANNOTATION_WRAP, break_long_words=False)
    return wrapped.replace(_NO_BREAK_SPACE, " ")


@dataclass(frozen=True, slots=True)
class ColumnAnnotation:
    """The text above a method's bar, standing on ``base``, the top of the column's last mark."""

    column: int
    base: float
    text: str

    @property
    def wrapped(self) -> str:
        """Each line of the text broken to about a bar's column width; line breaks are kept."""
        return "\n".join(_wrap_line(line) for line in self.text.split("\n"))

    @property
    def line_count(self) -> int:
        """How many lines the wrapped text takes."""
        return self.wrapped.count("\n") + 1


def value_axis_top(annotations: Sequence[ColumnAnnotation]) -> float | None:
    """The value-axis maximum that leaves every wrapped annotation room above its own base.

    An annotation of ``n`` lines needs ``n + 0.5`` line fractions of the axis above its base, so
    the axis top is the largest ``base / (1 - needed)``: a tall note over a short bar does not
    stretch the axis for the others. ``None`` when no mark rises above zero, so there is no scale
    to leave room on.
    """
    if max((annotation.base for annotation in annotations), default=0.0) <= 0.0:
        return None
    return max(
        annotation.base
        / (
            1.0
            - min(_MAX_ANNOTATION_FRACTION, _COLUMN_LINE_FRACTION * (annotation.line_count + 0.5))
        )
        for annotation in annotations
    )


def annotate_columns(ax: Axes, annotations: Sequence[ColumnAnnotation]) -> None:
    """Write each annotation above its bar and extend the value axis from 0 so all fit inside."""
    for annotation in annotations:
        ax.annotate(
            annotation.wrapped,
            (annotation.column, annotation.base),
            xytext=(0.0, _COLUMN_ANNOTATION_GAP_PT),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONT_SIZE,
        )
    top = value_axis_top(annotations)
    if top is None:
        ax.set_ylim(bottom=0.0)
    else:
        ax.set_ylim(0.0, top)
    ax.grid(True, axis="y", alpha=0.3)
