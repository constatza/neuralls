"""Render datasets, methods, and titles: the compositions that know what a dataset is."""

from __future__ import annotations

from dataclasses import replace
from typing import assert_never

from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    DatasetQualifiedMethodPresentation,
    StandaloneMethodPresentation,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.qualifier import dataset_qualifier
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
)
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import (
    NotationPolicy,
    RenderProfile,
    RenderTarget,
)
from neuralls.shared.scientific_description.render import RenderedScientificDescription, render

_QUALIFIER_SEPARATOR = " | "
_STRATEGY_SEPARATOR = " + "


def _render_chain(
    parts: tuple[ScientificDescription, ...], profile: RenderProfile, policy: NotationPolicy
) -> RenderedScientificDescription:
    """Render ``head | first-qualifier + next-qualifier ...`` with merged layout hints."""
    unbounded = replace(profile, max_length=None)
    plain = replace(policy, target=RenderTarget.PLAIN)
    head, *tail = (render(part, unbounded, policy) for part in parts)
    plain_head, *plain_tail = (render(part, unbounded, plain).text for part in parts)
    visible = plain_head + (_QUALIFIER_SEPARATOR if plain_tail else "")
    visible += _STRATEGY_SEPARATOR.join(plain_tail)
    text = head.text
    boundaries = list(head.preferred_wrap_boundaries)
    for index, part in enumerate(tail):
        separator = _QUALIFIER_SEPARATOR if index == 0 else _STRATEGY_SEPARATOR
        start = len(text) + len(separator)
        boundaries.append(start)
        boundaries.extend(start + offset for offset in part.preferred_wrap_boundaries)
        text += separator + part.text
    should_wrap = profile.max_length is not None and len(visible) > profile.max_length
    return RenderedScientificDescription(
        text=text, preferred_wrap_boundaries=tuple(boundaries) if should_wrap else ()
    )


def render_method(
    configured: ConfiguredMethodPresentation,
    profile: RenderProfile,
    policy: NotationPolicy,
) -> RenderedScientificDescription:
    """Render a method, appending its training dataset exactly once when it has one."""
    match configured:
        case StandaloneMethodPresentation(method=method):
            return render(method, profile, policy)
        case DatasetQualifiedMethodPresentation(method=method, dataset=dataset):
            return _render_chain((method, *dataset_qualifier(dataset)), profile, policy)
        case _:
            assert_never(configured)


def render_dataset(
    dataset: DatasetPresentation, profile: RenderProfile, policy: NotationPolicy
) -> RenderedScientificDescription:
    """Render a dataset as its source problem followed by each generation strategy."""
    return _render_chain((dataset.source.source, *dataset_qualifier(dataset)), profile, policy)


def render_title(
    title: ComparisonTitle, profile: RenderProfile, policy: NotationPolicy
) -> RenderedScientificDescription:
    """Render a case/comparison title: an editorial name, or the dataset it ran on."""
    match title:
        case EditorialPresentation(title=text):
            return RenderedScientificDescription(text=text)
        case ComparisonScopePresentation(dataset=dataset):
            return render_dataset(dataset, profile, policy)
        case _:
            assert_never(title)
