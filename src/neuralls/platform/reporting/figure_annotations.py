"""The count text and per-row annotations that report figures write beside their marks."""

from __future__ import annotations

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
