"""Drawing of a stored method's convergence-history figures."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import polars as pl

from neuralls.domain.presentation.methods import (
    MissingMethodPresentationError,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.recipes import render_recipe
from neuralls.domain.presentation.render import render_dataset
from neuralls.domain.solver.models.figure_scope import (
    MethodScope,
)
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.domain.solver.true_residual import shared_threshold, stopping_threshold_rel
from neuralls.platform.reporting.convergence_plots import (
    plot_convergence_comparison,
    plot_error_convergence_comparison,
)
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_metadata import (
    FigureKind,
)
from neuralls.platform.reporting.figure_recipes import history_variant
from neuralls.platform.reporting.figure_style import paper_style
from neuralls.platform.reporting.figure_text import (
    EditorialTitle,
    Legend,
    ScopeSubtitle,
    series_legend,
)
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.shared.rhs_recipe import RecipeKey, recipe_key
from neuralls.shared.scientific_description.policy import LEGEND_PROFILE, NotationPolicy
from neuralls.shared.types import DatasetId


class MethodPlot(Protocol):
    """A convergence-history plot of one stored method, one curve per attempted system."""

    def __call__(
        self,
        results: Mapping[str, Mapping[str, object]],
        *,
        target: FigureTarget,
        threshold_rel: float | None,
        title: EditorialTitle,
        subtitle: ScopeSubtitle,
        legend: Legend,
        group_for_system: Mapping[str, str],
        families: Mapping[str, Hashable],
        color_keys: Mapping[str, Hashable],
        show_titles: bool,
    ) -> None: ...


METHOD_PLOTS: Mapping[FigureKind, MethodPlot] = {
    FigureKind.RESIDUAL_HISTORY: plot_convergence_comparison,
    FigureKind.ENERGY_HISTORY: plot_error_convergence_comparison,
}
"""The plot of each method figure kind; a group kind has no entry and cannot be drawn here."""


@dataclass(frozen=True, slots=True)
class HistoryLegend:
    """One visible legend entry per dataset and RHS recipe group."""

    legend: Legend
    group_for_system: Mapping[str, str]


def _history_legend(
    systems: Sequence[SystemProvenance],
    plotted_ids: Collection[str],
    notation: NotationPolicy,
    datasets: Mapping[DatasetId, DatasetPresentation],
) -> HistoryLegend:
    """Name the distinct dataset/recipe groups of the histories being drawn."""
    selected = set(plotted_ids)
    groups: dict[tuple[DatasetId, RecipeKey], SystemProvenance] = {}
    group_for_system: dict[str, str] = {}
    for system in systems:
        if str(system.tag.system_id) not in selected:
            continue
        tag = system.tag
        group = (tag.dataset, recipe_key(tag.recipe))
        groups.setdefault(group, system)
        group_for_system[str(tag.system_id)] = str(groups[group].tag.system_id)

    show_dataset = len({dataset for dataset, _ in groups}) > 1
    show_recipe = len({recipe for _, recipe in groups}) > 1 or not show_dataset
    texts: dict[str, str] = {}
    for (dataset_id, _), system in groups.items():
        dataset = datasets.get(dataset_id)
        if dataset is None:
            raise MissingMethodPresentationError(str(dataset_id))
        parts = (
            render_dataset(dataset, LEGEND_PROFILE, notation).text if show_dataset else "",
            render_recipe(system.tag.recipe, LEGEND_PROFILE, notation) if show_recipe else "",
        )
        texts[str(system.tag.system_id)] = " | ".join(part for part in parts if part)

    # Distinct stored sources can have the same human-facing recipe text.
    by_text: dict[str, list[str]] = {}
    for group_id, label in texts.items():
        by_text.setdefault(label, []).append(group_id)
    for label, colliding in by_text.items():
        if len(colliding) > 1:
            for index, group_id in enumerate(colliding, start=1):
                texts[group_id] = f"{label}, variant {index}"
    return HistoryLegend(series_legend(texts), group_for_system)


def residual_threshold(
    solve_attempts: pl.DataFrame, systems: Sequence[SystemProvenance], system_ids: Collection[str]
) -> float | None:
    """The stopping threshold shared by every plotted system, or ``None`` when unknown or mixed.

    A system without a measured RHS norm has no threshold, so the line is then omitted rather
    than drawn for only some of the curves.
    """
    rhs_norms = dict(
        zip(
            solve_attempts.get_column("system_id").to_list(),
            solve_attempts.get_column("rhs_norm").to_list(),
            strict=True,
        )
    )
    solver = {str(system.tag.system_id): system.solver for system in systems}
    thresholds: list[float] = []
    for system_id in system_ids:
        rhs_norm = rhs_norms.get(system_id)
        if rhs_norm is None or rhs_norm <= 0:
            return None
        thresholds.append(
            stopping_threshold_rel(solver[system_id].rtol, solver[system_id].atol, rhs_norm)
        )
    return shared_threshold(thresholds)


def draw_method_figure(
    target: FigureTarget,
    kind: FigureKind,
    method: StoredMethod,
    notation: NotationPolicy,
    datasets: Mapping[DatasetId, DatasetPresentation],
    *,
    show_titles: bool,
) -> None:
    """Draw one method's residual or energy histories, one curve per attempted system."""
    converged = frozenset(
        method.solve_attempts.filter(pl.col("solve_status") == "converged")
        .get_column("system_id")
        .to_list()
    )
    histories: dict[str, dict[str, object]] = {}
    history_key = (
        "residual_history_rel" if kind is FigureKind.RESIDUAL_HISTORY else "error_history_a_rel"
    )
    for (system_id,), rows in method.convergence.sort("iteration").group_by(
        "system_id", maintain_order=True
    ):
        history: dict[str, object] = {
            "residual_history_rel": rows.get_column("residual_rel").to_list(),
            "error_history_a_rel": rows.get_column("energy_error_rel").drop_nulls().to_list(),
            "converged": str(system_id) in converged,
        }
        if history[history_key]:
            histories[str(system_id)] = history
    subtitle = ScopeSubtitle(MethodScope(method.presentation.prepared.detailed), notation)
    recipes = {
        str(system.tag.system_id): recipe_key(system.tag.recipe) for system in method.systems
    }
    series_recipes = {name: recipes[name] for name in histories}
    tags = {str(system.tag.system_id): system.tag for system in method.systems}
    color_keys = {name: (tags[name].dataset, recipes[name]) for name in histories}
    history_legend = _history_legend(method.systems, histories, notation, datasets)
    threshold = (
        residual_threshold(method.solve_attempts, method.systems, histories)
        if kind is FigureKind.RESIDUAL_HISTORY
        else None
    )
    with paper_style():
        METHOD_PLOTS[kind](
            histories,
            threshold_rel=threshold,
            target=target,
            title=EditorialTitle(history_variant(kind).title),
            subtitle=subtitle,
            legend=history_legend.legend,
            group_for_system=history_legend.group_for_system,
            families=series_recipes,
            color_keys=color_keys,
            show_titles=show_titles,
        )
