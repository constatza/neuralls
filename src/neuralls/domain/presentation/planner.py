"""Pure transitions between method-presentation states.

``prepare_presentations`` turns a configured cohort into the only state that
execution and tracking accept; ``reduce_realized_facts`` reduces the facts captured on
each system after setup. Both operate on structured facts and never render text.
"""

from __future__ import annotations

from collections.abc import Mapping

from neuralls.domain.presentation.methods import (
    MethodPresentationCandidate,
    PreparedComparisonPresentation,
    PreparedMethodPresentation,
    SystemRealization,
    UnvalidatedComparisonPresentation,
    description_facts,
)
from neuralls.shared.scientific_description.catalog import (
    FactVisibility,
    fact_definition,
    is_compact_by_policy,
)
from neuralls.shared.scientific_description.models import FieldKey, ScientificFact
from neuralls.shared.types import SystemId

__all__ = [
    "prepare_presentations",
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


def reduce_realized_facts(
    systems: tuple[SystemRealization, ...], population: frozenset[SystemId]
) -> tuple[ScientificFact, ...]:
    """Keep only facts with the same value on every system of ``population``.

    A fact is invariant over a population only if every one of its systems was realized;
    if any system has no realization (its setup failed), nothing is known to hold for the
    whole population, so no fact is kept.
    """
    if not systems or {system.system_id for system in systems} != population:
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
    description = candidate.presentation.method
    peers = tuple(
        other.presentation.method
        for other in cohort
        if other is not candidate and other.presentation.method.base_name == description.base_name
    )
    return tuple(
        fact
        for fact in description_facts(description)
        if _is_compact(fact, tuple(description_facts(peer) for peer in peers))
    )


def _is_compact(fact: ScientificFact, peer_facts: tuple[tuple[ScientificFact, ...], ...]) -> bool:
    visibility = fact_definition(fact.key).visibility_policy
    if is_compact_by_policy(visibility):
        return True
    return visibility is FactVisibility.WHEN_DISTINGUISHING and any(
        _value_of(peer, fact.key) != fact.value for peer in peer_facts
    )


def _value_of(facts: tuple[ScientificFact, ...], key: FieldKey) -> object | None:
    values: Mapping[FieldKey, object] = {fact.key: fact.value for fact in facts}
    return values.get(key)
