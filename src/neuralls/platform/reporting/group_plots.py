"""Drawing of the group outcome plots from their pre-draw figure plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from neuralls.domain.solver.models.group_outcomes import GroupReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.plot_style import plot_style_maps
from neuralls.platform.reporting.figure_metadata import (
    FigurePlan,
    GroupFigureMetadata,
    StagedFigure,
    StagedTabledFigure,
)
from neuralls.platform.reporting.figure_planning import group_recipe
from neuralls.platform.reporting.figure_text import (
    method_legend,
)
from neuralls.platform.reporting.group_drawers import FigureData
from neuralls.platform.reporting.outcome_plots import GroupStyle
from neuralls.shared.scientific_description.policy import NotationPolicy


@dataclass(frozen=True, slots=True)
class GroupStyler:
    """Turns the pre-draw plan of a figure into the style its plot draws with.

    The plan is the only source of a figure's fixed title and scope; the legend names
    every method of the group from its structured presentation.
    """

    presentation: GroupPresentation
    notation: NotationPolicy

    def style(self, plan: FigurePlan) -> GroupStyle:
        """The style for ``plan``: fixed title, rendered scope, legend and series styling."""
        maps = plot_style_maps(self.presentation.plot_styles)
        return GroupStyle(
            title=plan.title,
            scope=plan.metadata.scope,
            notation=self.notation,
            legend=method_legend(
                {name: method.compact for name, method in self.presentation.methods.items()},
                self.notation,
            ),
            families=maps.families,
            color_keys=maps.color_keys,
            marker_keys=maps.marker_keys,
        )


def write_outcome_plots(
    outcome: GroupReport,
    summaries: Mapping[str, pl.DataFrame],
    styler: GroupStyler,
    directory: Path,
    plans: Sequence[FigurePlan[GroupFigureMetadata]],
) -> tuple[StagedTabledFigure, ...]:
    """Draw every planned figure that has a summary and whose data supports it, in plan order.

    ``summaries`` holds the frame of each variant by stem; a drawn figure is staged with the
    frame it drew as its table.
    """
    drawn: list[StagedTabledFigure] = []
    for plan in plans:
        metadata = plan.metadata
        recipe = group_recipe(metadata.kind)
        summary = summaries.get(recipe.variant(metadata.metric, metadata.population).stem)
        if summary is None:
            continue
        data = FigureData(
            outcome.observations,
            summary,
            styler.style(plan),
            metadata.metric,
            outcome.reference_selection,
        )
        path = recipe.draw(data, directory / plan.filename)
        if path is not None:
            drawn.append(StagedTabledFigure(StagedFigure(plan, path), summary))
    return tuple(drawn)
