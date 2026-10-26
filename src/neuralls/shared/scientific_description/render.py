"""Pure rendering of a ``ScientificDescription`` into presentation text.

Formats facts only; never executes numerical work, reads a file, or queries
a strategy/preconditioner/storage object to produce a label.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from neuralls.shared.scientific_description.catalog import (
    FactVisibility,
    UncontrolledTextFactError,
    fact_definition,
)
from neuralls.shared.scientific_description.models import (
    DecimalFact,
    FieldKey,
    IntegerFact,
    ScientificDescription,
    ScientificFact,
    TextFact,
)
from neuralls.shared.scientific_description.policy import (
    NotationPolicy,
    RenderProfile,
    RenderTarget,
)

_DEFAULT_DECIMAL_PRECISION = 3
_INLINE_PROFILE_NAME = "_inline"


@dataclass(frozen=True, slots=True)
class RenderedScientificDescription:
    """Rendered text plus safe boundaries a sink may use for wrapping."""

    text: str
    preferred_wrap_boundaries: tuple[int, ...] = ()


def render(
    description: ScientificDescription,
    profile: RenderProfile,
    policy: NotationPolicy,
) -> RenderedScientificDescription:
    """Render all selected facts and return non-destructive layout hints."""
    selected = _select_facts(description, profile)
    visible_policy = replace(policy, target=RenderTarget.PLAIN)
    visible_text, _ = _compose(description.base_name, selected, profile, visible_policy)
    text, boundaries = _compose(description.base_name, selected, profile, policy)
    should_wrap = profile.max_length is not None and len(visible_text) > profile.max_length
    return RenderedScientificDescription(
        text=text,
        preferred_wrap_boundaries=boundaries if should_wrap else (),
    )


def render_facts(
    facts: tuple[ScientificFact, ...],
    policy: NotationPolicy,
    *,
    precision: Mapping[FieldKey, int] | None = None,
) -> str:
    """Render a flat ``"symbol=value, symbol=value"`` string, no base name or length budget.

    For callers that compose their own base name and surrounding text
    themselves (e.g. a preconditioner label combining a family abbreviation
    with structural detail) rather than using ``render()``'s
    ``ScientificDescription``-shaped ``base_name (facts)`` output.
    """
    profile = RenderProfile(
        name=_INLINE_PROFILE_NAME, include_context=False, precision=precision or {}
    )
    return ", ".join(
        _format_fact(fact, profile, policy) for fact in facts if _is_absolutely_visible(fact)
    )


def _select_facts(
    description: ScientificDescription, profile: RenderProfile
) -> list[ScientificFact]:
    facts = [fact for fact in description.facts if _is_absolutely_visible(fact)]
    facts.extend(
        fact
        for fact in description.context_facts
        if _is_absolutely_visible(fact)
        and (
            profile.include_context
            or fact_definition(fact.key).visibility_policy is FactVisibility.ALWAYS
        )
    )
    if profile.field_order is not None:
        order_index = {key: index for index, key in enumerate(profile.field_order)}
        facts.sort(key=lambda fact: order_index.get(fact.key, len(order_index)))
    return facts


def _is_absolutely_visible(fact: ScientificFact) -> bool:
    """Apply catalog policies that no rendering profile may override."""
    return fact_definition(fact.key).visibility_policy is not FactVisibility.NEVER


def _compose(
    base_name: str,
    facts: list[ScientificFact],
    profile: RenderProfile,
    policy: NotationPolicy,
) -> tuple[str, tuple[int, ...]]:
    if not facts:
        return base_name, ()
    formatted = tuple(_format_fact(fact, profile, policy) for fact in facts)
    prefix = f"{base_name} ("
    cursor = len(prefix)
    boundaries: list[int] = []
    for fragment in formatted[:-1]:
        cursor += len(fragment) + len(", ")
        boundaries.append(cursor)
    return f"{prefix}{', '.join(formatted)})", tuple(boundaries)


def _format_fact(fact: ScientificFact, profile: RenderProfile, policy: NotationPolicy) -> str:
    definition = fact_definition(fact.key)
    symbol = policy.symbol_for(fact.key)
    precision = profile.precision.get(fact.key, definition.precision)
    value_str = _format_value(fact, precision, policy.target)
    if policy.target is RenderTarget.PLAIN:
        return f"{symbol.plain}={value_str}"
    return f"${symbol.math}={value_str}$"


def _format_value(
    fact: ScientificFact,
    precision: int | None,
    target: RenderTarget,
) -> str:
    match fact:
        case IntegerFact(value=value):
            return str(value)
        case DecimalFact(value=value):
            digits = precision if precision is not None else _DEFAULT_DECIMAL_PRECISION
            return f"{value:.{digits}g}"
        case TextFact(value=value):
            formatter = fact_definition(fact.key).categorical_formatter
            if formatter is None:
                raise UncontrolledTextFactError(fact.key)
            if target is RenderTarget.PLAIN:
                return formatter.plain(fact.key, value)
            return formatter.mathtext(fact.key, value)
