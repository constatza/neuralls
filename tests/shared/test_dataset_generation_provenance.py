"""Dataset generation provenance holds one realized row count per strategy, and states the total."""

from __future__ import annotations

import pytest

from neuralls.shared.scientific_description import (
    LEGEND_PROFILE,
    DatasetGenerationProvenance,
    DatasetPresentation,
    DatasetSourceProvenance,
    DerivedSeed,
    ExplicitSeed,
    FieldImportance,
    FieldKey,
    IntegerFact,
    NotationPolicy,
    ScientificDescription,
    StrategyProvenance,
    render_dataset,
)
from neuralls.shared.scientific_description.catalog import (
    ITERATION_COUNT,
    KRYLOV_DIMENSION,
    SAMPLE_COUNT,
    SEED,
)
from neuralls.shared.types import GenerationStrategyKind

ESSENTIAL = FieldImportance.ESSENTIAL
POLICY = NotationPolicy.default()


def _strategy(kind: GenerationStrategyKind, rows: int) -> StrategyProvenance:
    return StrategyProvenance(kind, rows=rows, parameters=(), seed=DerivedSeed())


def _generation(
    *strategies: StrategyProvenance, rows: int | None = None
) -> DatasetGenerationProvenance:
    total = sum(strategy.rows for strategy in strategies) if rows is None else rows
    return DatasetGenerationProvenance(
        strategies=strategies, generated_rows=total, normalization="matrix", root_seed=1
    )


def _dataset(*strategies: StrategyProvenance) -> DatasetPresentation:
    return DatasetPresentation(
        DatasetSourceProvenance(ScientificDescription("Spectral 3D field")),
        _generation(*strategies),
    )


def test_strategy_rows_must_add_up_to_the_dataset_rows() -> None:
    """Kills: a total that is stored independently of the per-strategy rows."""
    with pytest.raises(ValueError, match="sum to 7 but the dataset has 8"):
        _generation(_strategy(GenerationStrategyKind.NORMAL, 7), rows=8)


def test_a_strategy_kind_may_appear_once() -> None:
    with pytest.raises(ValueError, match="unique"):
        _generation(
            _strategy(GenerationStrategyKind.NORMAL, 3), _strategy(GenerationStrategyKind.NORMAL, 4)
        )


@pytest.mark.parametrize("rows", [0, -1])
def test_a_strategy_must_have_produced_rows(rows: int) -> None:
    """A strategy that wrote nothing is absent from the provenance, never recorded as 0 or -1."""
    with pytest.raises(ValueError, match="positive"):
        _strategy(GenerationStrategyKind.NORMAL, rows)


@pytest.mark.parametrize("key", [SAMPLE_COUNT, SEED])
def test_the_count_and_the_seed_cannot_be_stored_as_parameters(key: FieldKey) -> None:
    """Kills: storing N or the seed twice, where the copies could disagree."""
    with pytest.raises(ValueError, match="must not hold"):
        StrategyProvenance(
            GenerationStrategyKind.NORMAL,
            rows=3,
            parameters=(IntegerFact(key, 1, ESSENTIAL),),
            seed=DerivedSeed(),
        )


def test_the_count_fact_is_derived_from_rows_after_the_leading_trajectory_cap() -> None:
    trace = StrategyProvenance(
        GenerationStrategyKind.GAUSSIAN_RESIDUALS,
        rows=5000,
        parameters=(IntegerFact(ITERATION_COUNT, 20, ESSENTIAL),),
        seed=ExplicitSeed(4),
    )
    assert [fact.key for fact in trace.facts] == [ITERATION_COUNT, SAMPLE_COUNT, SEED]

    krylov = StrategyProvenance(
        GenerationStrategyKind.KRYLOV,
        rows=3,
        parameters=(IntegerFact(KRYLOV_DIMENSION, 2, ESSENTIAL),),
        seed=DerivedSeed(),
    )
    assert [fact.key for fact in krylov.facts] == [KRYLOV_DIMENSION, SAMPLE_COUNT]
    assert next(f for f in krylov.facts if f.key == SAMPLE_COUNT).value == 3


def test_a_single_strategy_dataset_states_its_total_as_that_strategys_count() -> None:
    text = render_dataset(
        _dataset(_strategy(GenerationStrategyKind.NORMAL, 300)), LEGEND_PROFILE, POLICY
    ).text
    assert text == "Spectral 3D field | Random Normal Solutions (N=300)"


def test_a_mixed_dataset_states_the_total_and_every_strategys_count() -> None:
    """Kills: presenting only per-strategy counts for a mixture."""
    text = render_dataset(
        _dataset(
            _strategy(GenerationStrategyKind.NORMAL, 300),
            _strategy(GenerationStrategyKind.NEUTRAL_ONES, 100),
        ),
        LEGEND_PROFILE,
        POLICY,
    ).text
    assert text == (
        "Spectral 3D field | Mixture (N=400) + Random Normal Solutions (N=300)"
        " + Ones Solutions (N=100)"
    )
