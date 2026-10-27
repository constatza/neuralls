"""Fact-key uniqueness, compact-fact ordering, and typed identity in presentation carriers."""

from __future__ import annotations

import pytest

from neuralls.domain.presentation.methods import (
    MethodPresentationCandidate,
    PreparedMethodPresentation,
    StandaloneMethodPresentation,
    SystemRealization,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations
from neuralls.domain.presentation.provenance import (
    DatasetPresentation,
    DerivedSeed,
    StrategyProvenance,
)
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.domain.solver.models.presentation import GroupMethodPresentation
from neuralls.shared.scientific_description.catalog import (
    COARSE_DIMENSION,
    ITERATION_COUNT,
    MULTIGRID_LEVELS,
    RANK,
    THRESHOLD,
    make_fact,
)
from neuralls.shared.scientific_description.models import (
    DuplicateFactKeyError,
    ScientificDescription,
    ScientificFact,
)
from neuralls.shared.types import GenerationStrategyKind, MatrixId, PreconditionerName, SystemId

LEVELS = make_fact(MULTIGRID_LEVELS, 3)
THETA = make_fact(THRESHOLD, 0.05)
ALPHA = make_fact(ITERATION_COUNT, 7)


@pytest.fixture
def scope(dataset_presentation: DatasetPresentation) -> ComparisonScopePresentation:
    """The comparison scope of the shared small dataset."""
    return ComparisonScopePresentation(dataset_presentation)


@pytest.fixture
def amg() -> StandaloneMethodPresentation:
    """A method configured with levels, then threshold."""
    return StandaloneMethodPresentation(ScientificDescription("AMG", facts=(LEVELS, THETA)))


def _prepared(
    configured: StandaloneMethodPresentation, compact: tuple[ScientificFact, ...]
) -> PreparedMethodPresentation:
    return PreparedMethodPresentation(PreconditionerName("amg"), configured, compact)


def test_compact_facts_may_be_an_ordered_subsequence(amg: StandaloneMethodPresentation) -> None:
    assert _prepared(amg, (THETA,)).compact_facts == (THETA,)
    assert _prepared(amg, (LEVELS, THETA)).compact_facts == (LEVELS, THETA)
    assert _prepared(amg, ()).compact_facts == ()


def test_compact_facts_in_a_different_order_than_configured_are_rejected(
    amg: StandaloneMethodPresentation,
) -> None:
    with pytest.raises(ValueError, match="configured order"):
        _prepared(amg, (THETA, LEVELS))


def test_compact_fact_that_is_not_configured_is_rejected(amg: StandaloneMethodPresentation) -> None:
    with pytest.raises(ValueError, match="not a configured fact"):
        _prepared(amg, (ALPHA,))


def test_compact_fact_with_the_configured_key_but_another_value_is_rejected(
    amg: StandaloneMethodPresentation,
) -> None:
    with pytest.raises(ValueError, match="not a configured fact"):
        _prepared(amg, (make_fact(MULTIGRID_LEVELS, 4),))


def test_compact_facts_cannot_repeat_a_key(amg: StandaloneMethodPresentation) -> None:
    with pytest.raises(DuplicateFactKeyError):
        _prepared(amg, (LEVELS, LEVELS))


def test_system_realization_rejects_a_repeated_key() -> None:
    fact = make_fact(COARSE_DIMENSION, 12)
    with pytest.raises(DuplicateFactKeyError):
        SystemRealization(SystemId("s"), MatrixId("m"), (fact, fact))


def test_strategy_provenance_rejects_repeated_parameters() -> None:
    fact = make_fact(ITERATION_COUNT, 20)
    with pytest.raises(DuplicateFactKeyError):
        StrategyProvenance(
            GenerationStrategyKind.RESIDUALS, rows=10, parameters=(fact, fact), seed=DerivedSeed()
        )


def test_group_presentation_rejects_repeated_invariant_facts(
    amg: StandaloneMethodPresentation,
) -> None:
    fact = make_fact(COARSE_DIMENSION, 12)
    with pytest.raises(DuplicateFactKeyError):
        GroupMethodPresentation(prepared=_prepared(amg, ()), invariant_facts=(fact, fact))


def test_a_rank_of_ten_and_a_rank_of_ten_point_zero_are_different_methods(
    scope: ComparisonScopePresentation,
) -> None:
    """Python equates 10 and 10.0; the cohort must not report them as identical labels."""

    def method(name: str, rank: float) -> MethodPresentationCandidate:
        description = ScientificDescription("POD-2G", facts=(make_fact(RANK, rank),))
        return MethodPresentationCandidate(
            PreconditionerName(name), StandaloneMethodPresentation(description)
        )

    cohort = UnvalidatedComparisonPresentation(
        methods=(method("count", 10), method("fraction", 10.0)), scope=scope
    )
    assert len(prepare_presentations(cohort).methods) == 2
