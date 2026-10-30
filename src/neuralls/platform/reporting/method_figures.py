"""Drawing of a stored method's convergence-history figures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import polars as pl

from neuralls.domain.presentation.methods import (
    MissingMethodPresentationError,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.figure_scope import (
    MethodScope,
    RecipeScope,
)
from neuralls.platform.reporting.figure_metadata import (
    FigureKind,
)
from neuralls.platform.reporting.figure_text import (
    EditorialTitle,
    Legend,
    ScopeSubtitle,
    series_legend,
)
from neuralls.platform.reporting.plots import (
    plot_convergence_comparison,
    plot_error_convergence_comparison,
)
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId


def _system_legend(
    method: StoredMethod,
    notation: NotationPolicy,
    datasets: Mapping[DatasetId, DatasetPresentation],
) -> Legend:
    """Name each system of a stored method by its dataset, recipe, matrix and RHS position."""
    texts: dict[str, str] = {}
    for system in method.systems:
        tag = system.tag
        dataset = datasets.get(tag.dataset)
        if dataset is None:
            raise MissingMethodPresentationError(str(tag.dataset))
        where = ScopeSubtitle(RecipeScope(dataset, tag.recipe), notation).text
        texts[str(tag.system_id)] = f"{where}, matrix {int(tag.matrix_id)}, RHS {tag.rhs_ordinal}"
    return series_legend(texts)


def draw_method_figure(
    path: Path,
    kind: FigureKind,
    method: StoredMethod,
    notation: NotationPolicy,
    datasets: Mapping[DatasetId, DatasetPresentation],
) -> None:
    """Draw one method's residual or energy histories, one curve per attempted system."""
    converged = frozenset(
        method.solve_attempts.filter(pl.col("solve_status") == "converged")
        .get_column("system_id")
        .to_list()
    )
    histories: dict[str, dict[str, object]] = {}
    for (system_id,), rows in method.convergence.sort("iteration").group_by(
        "system_id", maintain_order=True
    ):
        histories[str(system_id)] = {
            "residual_history_rel": rows.get_column("residual_rel").to_list(),
            "error_history_a_rel": rows.get_column("energy_error_rel").drop_nulls().to_list(),
            "converged": str(system_id) in converged,
        }
    subtitle = ScopeSubtitle(MethodScope(method.presentation.prepared.detailed), notation)
    recipes = {
        str(system.tag.system_id): recipe_key(system.tag.recipe) for system in method.systems
    }
    series_recipes = {name: recipes[name] for name in histories}
    match kind:
        case FigureKind.RESIDUAL_HISTORY:
            plot, title = plot_convergence_comparison, "Residual History"
        case FigureKind.ENERGY_HISTORY:
            plot, title = plot_error_convergence_comparison, "Energy History"
        case _:
            raise ValueError(f"{kind} is not a method figure")
    plot(
        histories,
        save_path=path,
        title=EditorialTitle(title),
        subtitle=subtitle,
        legend=_system_legend(method, notation, datasets),
        families=series_recipes,
        color_keys=series_recipes,
    )
