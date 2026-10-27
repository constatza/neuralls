"""What a report figure is about, as typed values rendered only at the sink."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from neuralls.domain.presentation.methods import ConfiguredMethodPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.recipes import render_recipe
from neuralls.domain.presentation.render import render_dataset, render_method
from neuralls.shared.rhs_recipe import RhsRecipe
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderProfile
from neuralls.shared.scientific_description.render import RenderedScientificDescription

_ALL_DATASETS_TEXT = "All datasets"
_RECIPE_SEPARATOR = " - "
_METHOD_PREFIX = "Method "


@dataclass(frozen=True, slots=True)
class RecipeScope:
    """One dataset and one right-hand-side recipe: a homogeneous problem family."""

    dataset: DatasetPresentation
    recipe: RhsRecipe


@dataclass(frozen=True, slots=True)
class DatasetScope:
    """One dataset, pooling every recipe."""

    dataset: DatasetPresentation


@dataclass(frozen=True, slots=True)
class AllDatasetsScope:
    """Every dataset and recipe of the case."""


@dataclass(frozen=True, slots=True)
class MethodScope:
    """One method over every system it ran on."""

    method: ConfiguredMethodPresentation


type GroupScope = RecipeScope | DatasetScope | AllDatasetsScope
"""What one report group pools; medium and global groups are mixtures."""

type FigureScope = GroupScope | MethodScope
"""What a figure is about."""


def render_figure_scope(
    scope: FigureScope, profile: RenderProfile, policy: NotationPolicy
) -> RenderedScientificDescription:
    """Render a scope: a dataset (with its recipe), every dataset, or a method."""
    match scope:
        case RecipeScope(dataset=dataset, recipe=recipe):
            rendered = render_dataset(dataset, profile, policy)
            return RenderedScientificDescription(
                text=f"{rendered.text}{_RECIPE_SEPARATOR}{render_recipe(recipe, profile, policy)}",
                preferred_wrap_boundaries=rendered.preferred_wrap_boundaries,
            )
        case DatasetScope(dataset=dataset):
            return render_dataset(dataset, profile, policy)
        case AllDatasetsScope():
            return RenderedScientificDescription(text=_ALL_DATASETS_TEXT)
        case MethodScope(method=method):
            rendered = render_method(method, profile, policy)
            return RenderedScientificDescription(
                text=f"{_METHOD_PREFIX}{rendered.text}",
                preferred_wrap_boundaries=tuple(
                    len(_METHOD_PREFIX) + offset for offset in rendered.preferred_wrap_boundaries
                ),
            )
        case _:
            assert_never(scope)
