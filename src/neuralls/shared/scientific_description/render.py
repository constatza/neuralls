"""Pure rendering of a ``ScientificDescription`` into presentation text.

Formats facts only; never executes numerical work, reads a file, or queries
a strategy/preconditioner/storage object to produce a label.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

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


def render(
    description: ScientificDescription,
    profile: RenderProfile,
    policy: NotationPolicy,
) -> str:
    """Render ``description`` under ``profile`` using ``policy``'s notation."""
    selected = _select_facts(description, profile)
    # The budget is about what a reader sees, so it is measured on the plain
    # spelling for every target: `$\\theta=0.25$` is markup, not extra width.
    visible_policy = replace(policy, target=RenderTarget.PLAIN)
    while profile.max_length is not None and (
        len(_compose(description.base_name, selected, profile, visible_policy)) > profile.max_length
    ):
        droppable = [fact for fact in selected if _drop_priority(fact, description) > 0]
        if not droppable:
            break
        to_drop = max(droppable, key=lambda fact: _drop_priority(fact, description))
        selected = [fact for fact in selected if fact is not to_drop]
    return _compose(description.base_name, selected, profile, policy)


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
    return ", ".join(_format_fact(fact, profile, policy) for fact in facts)


def _select_facts(
    description: ScientificDescription, profile: RenderProfile
) -> list[ScientificFact]:
    facts = list(description.facts)
    if profile.include_context:
        facts += list(description.context_facts)
    if profile.field_order is not None:
        order_index = {key: index for index, key in enumerate(profile.field_order)}
        facts.sort(key=lambda fact: order_index.get(fact.key, len(order_index)))
    return facts


def _drop_priority(fact: ScientificFact, description: ScientificDescription) -> int:
    if fact in description.context_facts:
        return 2
    if fact.importance.value == "variant":
        return 1
    return 0


def _compose(
    base_name: str,
    facts: list[ScientificFact],
    profile: RenderProfile,
    policy: NotationPolicy,
) -> str:
    if not facts:
        return base_name
    formatted = ", ".join(_format_fact(fact, profile, policy) for fact in facts)
    return f"{base_name} ({formatted})"


_MATH_TEXT_SPECIALS = str.maketrans({"_": "\\_", "%": "\\%", "#": "\\#", "&": "\\&", "$": "\\$"})
"""Characters that break a mathtext/LaTeX fragment when they appear in a text value
(e.g. an underscore in a class name)."""


def _format_fact(fact: ScientificFact, profile: RenderProfile, policy: NotationPolicy) -> str:
    symbol = policy.symbol_for(fact.key)
    value_str = _format_value(fact, profile.precision.get(fact.key))
    if policy.target is RenderTarget.PLAIN:
        return f"{symbol.plain}={value_str}"
    if isinstance(fact, TextFact):
        value_str = value_str.translate(_MATH_TEXT_SPECIALS)
    return f"${symbol.math}={value_str}$"


def _format_value(fact: ScientificFact, precision: int | None) -> str:
    match fact:
        case IntegerFact(value=value):
            return str(value)
        case DecimalFact(value=value):
            digits = precision if precision is not None else _DEFAULT_DECIMAL_PRECISION
            return f"{value:.{digits}g}"
        case TextFact(value=value):
            return value
