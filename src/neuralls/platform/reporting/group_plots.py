"""Drawing of the group outcome plots from their pre-draw figure plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from neuralls.domain.solver.models.group_outcomes import GroupReport
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.plot_style import plot_style_maps
from neuralls.platform.config.models.figure_style import FigureStyleConfig
from neuralls.platform.reporting.distribution_mode import RowLimits
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_metadata import (
    FigurePlan,
    GroupFigureMetadata,
    StagedFigure,
    StagedTabledFigure,
)
from neuralls.platform.reporting.figure_recipes import group_recipe
from neuralls.platform.reporting.figure_style import paper_style
from neuralls.platform.reporting.figure_text import (
    method_legend,
)
from neuralls.platform.reporting.figure_writer import use_headless_backend
from neuralls.platform.reporting.group_drawers import FigureData
from neuralls.platform.reporting.group_style import GroupStyle
from neuralls.shared.scientific_description.policy import NotationPolicy


@dataclass(frozen=True, slots=True)
class GroupStyler:
    """Turns the pre-draw plan of a figure into the style its plot draws with.

    The plan is the only source of a figure's fixed title and scope; the legend names
    every method of the group from its structured presentation.
    """

    presentation: GroupPresentation
    notation: NotationPolicy
    figure_style: FigureStyleConfig = field(default_factory=FigureStyleConfig)

    def style(self, plan: FigurePlan) -> GroupStyle:
        """The style for ``plan``: fixed title, rendered scope, legend and series styling."""
        maps = plot_style_maps(self.presentation.plot_styles)
        return GroupStyle(
            title=plan.title,
            scope=plan.metadata.scope,
            cohort=plan.metadata.cohort,
            recipes=plan.metadata.recipes,
            n_matrix_sets=len(plan.metadata.dataset_ids),
            setup_sharing=plan.metadata.setup_sharing,
            notation=self.notation,
            legend=method_legend(
                {name: method.compact for name, method in self.presentation.methods.items()},
                self.notation,
            ),
            families=maps.families,
            color_keys=maps.color_keys,
            label_rotation_deg=self.figure_style.label_rotation_deg,
            row_limits=RowLimits(
                self.figure_style.min_box_samples, self.figure_style.min_violin_samples
            ),
            show_titles=self.figure_style.show_titles,
        )


def write_outcome_plots(
    outcome: GroupReport,
    summaries: Mapping[str, pl.DataFrame],
    styler: GroupStyler,
    directory: Path,
    plans: Sequence[FigurePlan[GroupFigureMetadata]],
) -> tuple[StagedTabledFigure, ...]:
    """Draw every planned figure whose data supports it, in plan order.

    ``summaries`` holds the frame of each planned variant by stem (``figure_summaries``); a figure
    draws it whole over the group's observations and is staged with it as its table.
    """
    use_headless_backend()
    drawn: list[StagedTabledFigure] = []
    for plan in plans:
        metadata = plan.metadata
        recipe = group_recipe(metadata.kind)
        summary = summaries[recipe.variant(metadata.metric, metadata.population).stem]
        data = FigureData(
            outcome.observations,
            summary,
            styler.style(plan),
            metadata.metric,
            outcome.reference_selection,
        )
        with paper_style():
            target = recipe.draw(data, FigureTarget(directory, plan.stem))
        if target is not None:
            drawn.append(StagedTabledFigure(StagedFigure(plan, target), summary))
    return tuple(drawn)
