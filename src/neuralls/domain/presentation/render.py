"""Render datasets, methods, and titles: the compositions that know what a dataset is."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from neuralls.shared.scientific_description.catalog import fact_definition
from neuralls.shared.scientific_description.models import (
    DecimalFact,
    FieldKey,
    ScientificDescription,
)
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


_MAX_EXTRA_DIGITS = 14
"""Significant digits added beyond the catalog precision before values are called equal."""


class IndistinguishableMethodsError(ValueError):
    """Two methods with different structured facts still render to the same text.

    ``method_ids`` are diagnostic machine identifiers only; they never become labels.
    """

    def __init__(self, method_ids: tuple[str, ...]) -> None:
        self.method_ids = method_ids
        super().__init__(
            "Methods render identically even at full numeric precision "
            f"(method ids: {', '.join(method_ids)})"
        )


@dataclass(frozen=True, slots=True)
class DistinctRendering:
    """Pairwise-distinct renderings plus the profile that made them distinct."""

    profile: RenderProfile
    texts: Mapping[str, RenderedScientificDescription]


def _decimal_keys(configured: ConfiguredMethodPresentation) -> frozenset[FieldKey]:
    match configured:
        case StandaloneMethodPresentation(method=method):
            parts = (method,)
        case DatasetQualifiedMethodPresentation(method=method, dataset=dataset):
            parts = (method, *dataset_qualifier(dataset))
        case _:
            assert_never(configured)
    return frozenset(
        fact.key
        for part in parts
        for fact in (*part.facts, *part.context_facts)
        if isinstance(fact, DecimalFact)
    )


def render_distinct_methods(
    methods: Mapping[str, ConfiguredMethodPresentation],
    profile: RenderProfile,
    policy: NotationPolicy,
) -> DistinctRendering:
    """Render every method so that different methods never share a text.

    Methods whose texts collide only because decimals were rounded (a drop tolerance of
    0.0012 and 0.0014 at one digit) are re-rendered with more significant digits for
    the decimals they contain, until their texts differ. Methods that stay identical at
    full precision cannot be told apart by any rendering and raise.
    """
    keys: frozenset[FieldKey] = frozenset()
    current = profile
    for extra in range(_MAX_EXTRA_DIGITS + 1):
        rendered = {name: render_method(m, current, policy) for name, m in methods.items()}
        counts = Counter(item.text for item in rendered.values())
        colliding = tuple(name for name, item in rendered.items() if counts[item.text] > 1)
        if not colliding:
            return DistinctRendering(profile=current, texts=rendered)
        keys |= frozenset().union(*(_decimal_keys(methods[name]) for name in colliding))
        current = replace(
            profile,
            precision={
                **profile.precision,
                **{
                    key: profile.precision.get(key, fact_definition(key).precision or 1) + extra + 1
                    for key in keys
                },
            },
        )
    raise IndistinguishableMethodsError(colliding)
