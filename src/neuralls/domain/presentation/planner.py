"""Pure transitions between method-presentation states.

``prepare_presentations`` turns a configured cohort into the only state that
execution and tracking accept; ``realize_method`` attaches the facts captured on
each system after setup. Both operate on structured facts and never render text.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_never

from neuralls.domain.presentation.methods import (
    DuplicateMethodPresentationError,
    ExecutedMethodPresentation,
    MethodPresentationCandidate,
    PreparedComparisonPresentation,
    PreparedMethodPresentation,
    RealizedMethodPresentation,
    SystemRealization,
    UnvalidatedComparisonPresentation,
    description_facts,
    method_description,
)
from neuralls.shared.scientific_description.catalog import FactVisibility, fact_definition
from neuralls.shared.scientific_description.models import FieldKey, ScientificFact

__all__ = [
    "DuplicateMethodPresentationError",
    "prepare_presentations",
    "prepared_of",
    "realizations_of",
    "realize_method",
    "reduce_realized_facts",
]


def prepare_presentations(
    unvalidated: UnvalidatedComparisonPresentation,
) -> PreparedComparisonPresentation:
    """Choose compact-label facts per catalog policy and validate the whole cohort.

    Algorithmic facts are always compact. Reproducibility facts are compact only when
    their value differs between methods of the same family. Scheduling facts are
    compact when present, because describers emit them only when non-default.
    Raises ``DuplicateMethodPresentationError`` if two compact presentations coincide.
    """
    methods = tuple(
        PreparedMethodPresentation(
            method_id=candidate.method_id,
            configured=candidate.presentation,
            compact_facts=_compact_facts(candidate, unvalidated.methods),
        )
        for candidate in unvalidated.methods
    )
    return PreparedComparisonPresentation(methods=methods, scope=unvalidated.scope)


def realize_method(
    prepared: PreparedMethodPresentation, systems: tuple[SystemRealization, ...]
) -> RealizedMethodPresentation:
    """Attach per-system realized facts to a prepared method."""
    return RealizedMethodPresentation(prepared=prepared, systems=systems)


def prepared_of(executed: ExecutedMethodPresentation) -> PreparedMethodPresentation:
    """The prepared presentation underlying either execution outcome."""
    match executed:
        case RealizedMethodPresentation(prepared=prepared):
            return prepared
        case PreparedMethodPresentation() as prepared:
            return prepared
        case _ as unreachable:
            assert_never(unreachable)


def realizations_of(executed: ExecutedMethodPresentation) -> tuple[SystemRealization, ...]:
    """The per-system facts captured after setup; empty when no setup succeeded."""
    match executed:
        case RealizedMethodPresentation(systems=systems):
            return systems
        case PreparedMethodPresentation():
            return ()
        case _ as unreachable:
            assert_never(unreachable)


def reduce_realized_facts(
    systems: tuple[SystemRealization, ...],
) -> tuple[ScientificFact, ...]:
    """Keep only facts with the same value on every system of the selected population."""
    if not systems:
        return ()
    first, *rest = systems
    return tuple(
        fact
        for fact in first.facts
        if all(_value_of(other.facts, fact.key) == fact.value for other in rest)
    )


def _compact_facts(
    candidate: MethodPresentationCandidate,
    cohort: tuple[MethodPresentationCandidate, ...],
) -> tuple[ScientificFact, ...]:
    description = method_description(candidate.presentation)
    peers = tuple(
        method_description(other.presentation)
        for other in cohort
        if other is not candidate
        and method_description(other.presentation).base_name == description.base_name
    )
    return tuple(
        fact
        for fact in description_facts(description)
        if _is_compact(fact, tuple(description_facts(peer) for peer in peers))
    )


def _is_compact(fact: ScientificFact, peer_facts: tuple[tuple[ScientificFact, ...], ...]) -> bool:
    match fact_definition(fact.key).visibility_policy:
        case FactVisibility.ALWAYS | FactVisibility.WHEN_NON_DEFAULT:
            return True
        case FactVisibility.WHEN_DISTINGUISHING:
            return any(_value_of(peer, fact.key) != fact.value for peer in peer_facts)
        case FactVisibility.DETAILED_ONLY | FactVisibility.NEVER:
            return False


def _value_of(facts: tuple[ScientificFact, ...], key: FieldKey) -> object | None:
    values: Mapping[FieldKey, object] = {fact.key: fact.value for fact in facts}
    return values.get(key)
