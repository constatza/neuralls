"""Describe a dataset by the generation strategies that produced it."""

from __future__ import annotations

from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.strategy_names import strategy_base_name
from neuralls.shared.scientific_description.catalog import (
    SAMPLE_COUNT,
    FactVisibility,
    fact_definition,
    make_fact,
)
from neuralls.shared.scientific_description.models import (
    ScientificDescription,
)

_QUALIFIER_VISIBILITY = frozenset({FactVisibility.ALWAYS, FactVisibility.WHEN_NON_DEFAULT})


_MIXTURE_BASE_NAME = "Mixture"


def dataset_qualifier(dataset: DatasetPresentation) -> tuple[ScientificDescription, ...]:
    """Describe the dataset by its generation strategies, one description each.

    A single strategy's ``N`` is the dataset's total. When several strategies mix, a
    leading ``Mixture`` description carries the total so every dataset states how many
    samples it holds. Reproducibility facts such as seeds stay in the structured
    provenance only; a qualifier shows what generated the data, not how it was seeded.
    """
    generation = dataset.generation
    strategies = tuple(
        ScientificDescription(
            base_name=strategy_base_name(strategy.kind),
            facts=tuple(
                fact
                for fact in strategy.facts
                if fact_definition(fact.key).visibility_policy in _QUALIFIER_VISIBILITY
            ),
        )
        for strategy in generation.strategies
    )
    if len(strategies) == 1:
        return strategies
    total = make_fact(SAMPLE_COUNT, generation.generated_rows)
    return (ScientificDescription(_MIXTURE_BASE_NAME, facts=(total,)), *strategies)
