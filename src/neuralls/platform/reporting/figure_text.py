"""Text on a figure, rendered from typed presentation values at the drawing boundary.

Callers hand the plot functions typed values (method presentations, comparison titles,
scopes); this module turns them into renderer-owned text exactly when a figure is drawn.
The length limits control only where lines break and where the legend sits; no fact is
ever dropped to fit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal, assert_never

from matplotlib.axes import Axes

from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    MissingMethodPresentationError,
)
from neuralls.domain.presentation.render import render_distinct_methods, render_title
from neuralls.domain.presentation.run_names import TrainingPresentation
from neuralls.domain.presentation.titles import ComparisonTitle
from neuralls.domain.solver.models.figure_scope import FigureScope, render_figure_scope
from neuralls.platform.tracking.run_names import render_training_label
from neuralls.shared.scientific_description.policy import (
    LEGEND_PROFILE,
    LOG_PROFILE,
    NotationPolicy,
    RenderProfile,
    RenderTarget,
)
from neuralls.shared.scientific_description.render import RenderedScientificDescription

__all__ = [
    "ComparisonFigureTitle",
    "ConvergenceStatus",
    "EditorialTitle",
    "FigureTitle",
    "Legend",
    "LegendEntry",
    "LegendLayout",
    "LegendPlacement",
    "ScopeSubtitle",
    "draw_legend",
    "method_legend",
    "scope_label",
    "series_legend",
    "title_text",
    "training_legend",
    "wrap_at_boundaries",
]

TITLE_PROFILE = RenderProfile(name="title", include_context=False, max_length=90)
"""Titles and subtitles break lines past this many visible characters."""

DID_NOT_CONVERGE_MARK = "[did not converge]"
"""Suffix on a failed run's label, so a run stopped by the iteration cap is never read
as a cheap solve in the cost, iteration and time plots."""

MAX_INLINE_ENTRIES = 6
"""Most legend entries that stay inside the axes."""

MAX_LINES_PER_COLUMN = 24
"""Most legend text lines an external legend stacks before it opens another column."""

_LEGEND_ANCHOR = (1.02, 1.0)

LEGEND_FONT_SIZE = 7
"""Point size of every method legend entry and of the method labels on bar-chart axes."""


def wrap_at_boundaries(
    drawn: RenderedScientificDescription, visible: RenderedScientificDescription, limit: int
) -> str:
    """Break ``drawn`` into lines of at most ``limit`` visible characters at safe boundaries.

    ``visible`` is the same description rendered as plain text; its boundaries pair one to
    one with those of ``drawn`` and measure what a reader sees, markup excluded. A line
    that cannot be broken any further stays long: nothing is ever removed.
    """
    if not drawn.preferred_wrap_boundaries:
        return drawn.text
    if len(drawn.preferred_wrap_boundaries) != len(visible.preferred_wrap_boundaries):
        raise ValueError("drawn and visible renderings disagree on their wrap boundaries")
    drawn_cuts = (0, *drawn.preferred_wrap_boundaries, len(drawn.text))
    visible_cuts = (0, *visible.preferred_wrap_boundaries, len(visible.text))
    lines: list[str] = []
    line_start = 0
    for index in range(1, len(drawn_cuts) - 1):
        next_end = visible_cuts[index + 1]
        if next_end - visible_cuts[line_start] > limit:
            lines.append(drawn.text[drawn_cuts[line_start] : drawn_cuts[index]].rstrip())
            line_start = index
    lines.append(drawn.text[drawn_cuts[line_start] :])
    return "\n".join(lines)


def _plain(policy: NotationPolicy) -> NotationPolicy:
    return replace(policy, target=RenderTarget.PLAIN)


class ConvergenceStatus(StrEnum):
    """Whether the run behind a legend entry converged."""

    CONVERGED = "converged"
    DID_NOT_CONVERGE = "did_not_converge"


@dataclass(frozen=True, slots=True)
class LegendEntry:
    text: str
    convergence: ConvergenceStatus

    @property
    def drawn(self) -> str:
        match self.convergence:
            case ConvergenceStatus.CONVERGED:
                return self.text
            case ConvergenceStatus.DID_NOT_CONVERGE:
                return f"{self.text} {DID_NOT_CONVERGE_MARK}"
            case _:
                assert_never(self.convergence)


type InsideLegendLocation = Literal["upper right", "lower right"]
"""Where a legend that fits sits inside the axes."""


class LegendPlacement(StrEnum):
    """Where a legend is drawn relative to the axes."""

    INSIDE = "inside"
    OUTSIDE_RIGHT = "outside_right"


@dataclass(frozen=True, slots=True)
class LegendLayout:
    """How a legend is laid out: where it sits and across how many columns."""

    placement: LegendPlacement
    columns: int


@dataclass(frozen=True, slots=True)
class Legend:
    """The text of every series of a figure, keyed by series name.

    Built from typed presentations; asking for a series that has no entry raises, so an
    identifier can never stand in for a label.
    """

    entries: Mapping[str, LegendEntry]

    def label(self, name: str) -> str:
        """The drawn text of series ``name``, line breaks and convergence mark included."""
        return self._entry(name).drawn

    def marking_unconverged(self, names: Iterable[str]) -> Legend:
        """A legend whose entries for ``names`` carry the did-not-converge mark."""
        marked = {
            name: replace(self._entry(name), convergence=ConvergenceStatus.DID_NOT_CONVERGE)
            for name in names
        }
        return Legend({**self.entries, **marked})

    def _entry(self, name: str) -> LegendEntry:
        entry = self.entries.get(name)
        if entry is None:
            raise MissingMethodPresentationError(name)
        return entry

    @property
    def layout(self) -> LegendLayout:
        """Choose where the legend sits from the lines its entries need; never from fewer facts."""
        lines = sum(entry.drawn.count("\n") + 1 for entry in self.entries.values())
        if len(self.entries) <= MAX_INLINE_ENTRIES and lines == len(self.entries):
            return LegendLayout(LegendPlacement.INSIDE, columns=1)
        return LegendLayout(
            LegendPlacement.OUTSIDE_RIGHT, columns=max(1, math.ceil(lines / MAX_LINES_PER_COLUMN))
        )


def method_legend(
    methods: Mapping[str, ConfiguredMethodPresentation], policy: NotationPolicy
) -> Legend:
    """Render each method's compact presentation, distinct from every other, wrapped to fit."""
    limit = LEGEND_PROFILE.max_length
    distinct = render_distinct_methods(methods, LEGEND_PROFILE, policy)
    from_plain = render_distinct_methods(methods, distinct.profile, _plain(policy))
    entries = {
        name: LegendEntry(
            text=wrap_at_boundaries(drawn, from_plain.texts[name], limit)
            if limit is not None
            else drawn.text,
            convergence=ConvergenceStatus.CONVERGED,
        )
        for name, drawn in distinct.texts.items()
    }
    return Legend(entries)


def series_legend(texts: Mapping[str, str]) -> Legend:
    """A legend of series that are not methods, with text composed from typed numeric parts."""
    return Legend(
        {name: LegendEntry(text, ConvergenceStatus.CONVERGED) for name, text in texts.items()}
    )


def training_legend(presentations: Mapping[str, TrainingPresentation]) -> Legend:
    """Render what each assignment trains; the keys are the machine ids that order the bars."""
    return series_legend(
        {key: render_training_label(presentation) for key, presentation in presentations.items()}
    )


def scope_label(scope: FigureScope, notation: NotationPolicy) -> str:
    """The scope as one plain line, for logs and summaries."""
    return render_figure_scope(scope, LOG_PROFILE, _plain(notation)).text


def draw_legend(
    ax: Axes,
    legend: Legend,
    *,
    inside_loc: InsideLegendLocation,
    fontsize: float = LEGEND_FONT_SIZE,
) -> None:
    """Draw ``ax``'s legend where ``legend.layout`` says: inside, or outside in columns."""
    layout = legend.layout
    match layout.placement:
        case LegendPlacement.INSIDE:
            ax.legend(loc=inside_loc, fontsize=fontsize)
        case LegendPlacement.OUTSIDE_RIGHT:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=_LEGEND_ANCHOR,
                ncols=layout.columns,
                fontsize=fontsize,
            )
        case _:
            assert_never(layout.placement)


@dataclass(frozen=True, slots=True)
class EditorialTitle:
    """Fixed wording with no scientific content, such as ``Residual History``."""

    text: str


@dataclass(frozen=True, slots=True)
class ComparisonFigureTitle:
    """What was compared, and the size of the linear system it was compared on."""

    comparison: ComparisonTitle
    system_size: int | None
    notation: NotationPolicy


type FigureTitle = EditorialTitle | ComparisonFigureTitle


def _title_limit() -> int:
    limit = TITLE_PROFILE.max_length
    if limit is None:
        raise ValueError("TITLE_PROFILE must bound the line length")
    return limit


def title_text(title: FigureTitle) -> str:
    """The text above a figure."""
    match title:
        case EditorialTitle(text=text):
            return text
        case ComparisonFigureTitle(comparison=comparison, system_size=size, notation=notation):
            drawn = render_title(comparison, TITLE_PROFILE, notation)
            visible = render_title(comparison, TITLE_PROFILE, _plain(notation))
            text = wrap_at_boundaries(drawn, visible, _title_limit())
            return text if size is None else f"{text}\nN={size}"
        case _:
            assert_never(title)


@dataclass(frozen=True, slots=True)
class ScopeSubtitle:
    """The scope a figure is about, rendered under its metric title."""

    scope: FigureScope
    notation: NotationPolicy

    @property
    def text(self) -> str:
        """The scope as text, wrapped at safe boundaries."""
        drawn = render_figure_scope(self.scope, TITLE_PROFILE, self.notation)
        visible = render_figure_scope(self.scope, TITLE_PROFILE, _plain(self.notation))
        return wrap_at_boundaries(drawn, visible, _title_limit())
