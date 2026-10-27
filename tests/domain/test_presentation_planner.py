"""Presentation lifecycle: configured -> prepared -> realized, and cohort planning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from neuralls.domain.presentation.methods import (
    DatasetQualifiedMethodPresentation,
    DuplicateMethodPresentationError,
    ExecutedMethod,
    MethodPresentationCandidate,
    PreparedComparisonPresentation,
    PreparedMethodPresentation,
    StandaloneMethodPresentation,
    SystemRealization,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import (
    prepare_presentations,
    reduce_realized_facts,
)
from neuralls.domain.presentation.provenance import (
    DatasetGenerationProvenance,
    DatasetPresentation,
    DatasetSourceProvenance,
    DerivedSeed,
    StrategyProvenance,
)
from neuralls.domain.presentation.render import render_method
from neuralls.domain.presentation.titles import ComparisonScopePresentation, EditorialPresentation
from neuralls.shared.scientific_description.catalog import (
    ASSIGNMENT,
    CANDIDATE_COUNT,
    COARSE_DIMENSION,
    ITERATION_COUNT,
    MULTIGRID_LEVELS,
    RANK,
    SAMPLE_COUNT,
    SEED,
    START_ITERATION,
    SYSTEM_DIMENSION,
    THRESHOLD,
    make_fact,
)
from neuralls.shared.scientific_description.models import (
    FieldKey,
    ScientificDescription,
    ScientificFact,
)
from neuralls.shared.scientific_description.policy import LEGEND_PROFILE, NotationPolicy
from neuralls.shared.types import (
    GenerationStrategyKind,
    MatrixId,
    PreconditionerName,
    SystemId,
)


def _int(key: FieldKey, value: int) -> ScientificFact:
    return make_fact(key, value)


def _asa(
    *, seed: int = 0, candidates: int = 3, scheduling: tuple[ScientificFact, ...] = ()
) -> ScientificDescription:
    return ScientificDescription(
        base_name="aSA",
        facts=(
            _int(MULTIGRID_LEVELS, 2),
            _int(CANDIDATE_COUNT, candidates),
            make_fact(THRESHOLD, 0.05),
            _int(SEED, seed),
            *scheduling,
        ),
    )


def _candidate(name: str, method: ScientificDescription) -> MethodPresentationCandidate:
    return MethodPresentationCandidate(
        PreconditionerName(name), StandaloneMethodPresentation(method)
    )


@pytest.fixture
def dataset() -> DatasetPresentation:
    """A Gaussian-CG-generated dataset of a spectral 3D field."""
    return DatasetPresentation(
        source=DatasetSourceProvenance(ScientificDescription("Spectral 3D field")),
        generation=DatasetGenerationProvenance(
            strategies=(
                StrategyProvenance(
                    GenerationStrategyKind.GAUSSIAN_RESIDUALS,
                    rows=5000,
                    parameters=(_int(ITERATION_COUNT, 20),),
                    seed=DerivedSeed(),
                ),
            ),
            generated_rows=5000,
            normalization="matrix",
            root_seed=42,
        ),
    )


@pytest.fixture
def scope(dataset: DatasetPresentation) -> ComparisonScopePresentation:
    return ComparisonScopePresentation(dataset)


def _cohort(
    scope: ComparisonScopePresentation, *methods: MethodPresentationCandidate
) -> UnvalidatedComparisonPresentation:
    return UnvalidatedComparisonPresentation(methods=methods, scope=scope)


def _compact_keys(method: PreparedMethodPresentation) -> list[str]:
    return [fact.key.value for fact in method.compact_facts]


# --- immutable domain -------------------------------------------------------------


def test_presentation_values_are_frozen_and_slotted(dataset: DatasetPresentation) -> None:
    standalone = StandaloneMethodPresentation(_asa())
    qualified = DatasetQualifiedMethodPresentation(_asa(), dataset)
    for value in (standalone, qualified, dataset, EditorialPresentation("Spectral 3D")):
        assert is_dataclass(value)
        assert not hasattr(value, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(value, fields(value)[0].name, None)


def test_standalone_presentation_has_no_dataset_field() -> None:
    assert {f.name for f in fields(StandaloneMethodPresentation)} == {"method"}
    assert {f.name for f in fields(DatasetQualifiedMethodPresentation)} == {"method", "dataset"}


def test_dataset_generation_requires_a_strategy() -> None:
    with pytest.raises(ValueError, match="at least one strategy"):
        DatasetGenerationProvenance(
            strategies=(), generated_rows=1, normalization="none", root_seed=0
        )


@pytest.mark.parametrize("title", ["", " padded ", "two\nlines", "snake_case_title"])
def test_editorial_presentation_rejects_invalid_titles(title: str) -> None:
    with pytest.raises(ValueError):
        EditorialPresentation(title)


def test_assignment_identity_is_never_rendered(dataset: DatasetPresentation) -> None:
    method = ScientificDescription(
        "POD-2G",
        facts=(_int(RANK, 10), make_fact(ASSIGNMENT, 1)),
    )
    text = render_method(
        DatasetQualifiedMethodPresentation(method, dataset),
        LEGEND_PROFILE,
        NotationPolicy.default(),
    ).text
    assert "assignment" not in text
    assert "id=" not in text


def test_pod_method_composes_with_its_dataset_exactly_once(dataset: DatasetPresentation) -> None:
    method = ScientificDescription("POD-2G", facts=(_int(RANK, 10),))
    rendered = render_method(
        DatasetQualifiedMethodPresentation(method, dataset),
        LEGEND_PROFILE,
        NotationPolicy.default(),
    )
    assert rendered.text == "POD-2G (r=10) | Gaussian CG Residuals (k=20, N=5000)"
    assert rendered.text.count("|") == 1
    assert rendered.text.count("Gaussian CG Residuals") == 1


# --- cohort visibility ------------------------------------------------------------


def test_seed_is_compact_only_when_it_distinguishes_variants(
    scope: ComparisonScopePresentation,
) -> None:
    seeds = prepare_presentations(
        _cohort(scope, _candidate("a", _asa(seed=0)), _candidate("b", _asa(seed=1)))
    )
    candidates = prepare_presentations(
        _cohort(
            scope,
            _candidate("a", _asa(candidates=3)),
            _candidate("b", _asa(candidates=7)),
        )
    )

    assert all("seed" in _compact_keys(m) for m in seeds.methods)
    assert all("seed" not in _compact_keys(m) for m in candidates.methods)
    for cohort in (seeds, candidates):
        for method in cohort.methods:
            assert {"multigrid_levels", "candidate_count", "threshold"} <= set(
                _compact_keys(method)
            )
            assert "seed" in [fact.key.value for fact in method.detailed_facts]


def test_non_default_scheduling_facts_are_compact_and_defaults_are_absent(
    scope: ComparisonScopePresentation,
) -> None:
    scheduled = _asa(scheduling=(_int(START_ITERATION, 5),), candidates=3)
    prepared = prepare_presentations(
        _cohort(scope, _candidate("late", scheduled), _candidate("plain", _asa(candidates=7)))
    )
    by_id = {m.method_id: m for m in prepared.methods}

    assert "start_iteration" in _compact_keys(by_id[PreconditionerName("late")])
    assert "start_iteration" not in _compact_keys(by_id[PreconditionerName("plain")])


def test_detailed_only_and_never_facts_are_not_compact(scope: ComparisonScopePresentation) -> None:
    method = ScientificDescription(
        "Jacobi",
        facts=(_int(SYSTEM_DIMENSION, 45), make_fact(ASSIGNMENT, 1)),
    )
    prepared = prepare_presentations(_cohort(scope, _candidate("j", method)))
    only = prepared.methods[0]

    assert only.compact_facts == ()
    assert [f.key.value for f in only.detailed_facts] == ["system_dimension"]


def test_compact_facts_must_belong_to_the_configured_method(
    dataset: DatasetPresentation,
) -> None:
    with pytest.raises(ValueError, match="not a configured fact"):
        PreparedMethodPresentation(
            PreconditionerName("a"),
            StandaloneMethodPresentation(_asa()),
            (_int(RANK, 3),),
        )


# --- collisions -------------------------------------------------------------------


def test_semantic_duplicates_raise_before_rendering(scope: ComparisonScopePresentation) -> None:
    cohort = _cohort(scope, _candidate("first", _asa()), _candidate("second", _asa()))

    with pytest.raises(DuplicateMethodPresentationError) as caught:
        prepare_presentations(cohort)

    assert set(caught.value.method_ids) == {"first", "second"}
    assert "[" not in str(caught.value)


def test_prepared_cohort_cannot_be_constructed_around_validation(
    scope: ComparisonScopePresentation,
) -> None:
    method = PreparedMethodPresentation(
        PreconditionerName("x"), StandaloneMethodPresentation(_asa()), ()
    )
    twin = PreparedMethodPresentation(
        PreconditionerName("y"), StandaloneMethodPresentation(_asa()), ()
    )
    with pytest.raises(DuplicateMethodPresentationError):
        PreparedComparisonPresentation((method, twin), scope)
    with pytest.raises(ValueError, match="at least one method"):
        PreparedComparisonPresentation((), scope)
    with pytest.raises(ValueError, match="unique"):
        PreparedComparisonPresentation((method, method), scope)


def test_same_method_on_different_datasets_is_not_a_duplicate(
    dataset: DatasetPresentation, scope: ComparisonScopePresentation
) -> None:
    other = DatasetPresentation(
        dataset.source,
        DatasetGenerationProvenance(
            (
                StrategyProvenance(
                    GenerationStrategyKind.SMOOTHER_FILTERED_PROBES, 10, (), DerivedSeed()
                ),
            ),
            generated_rows=10,
            normalization="matrix",
            root_seed=42,
        ),
    )
    pod = ScientificDescription("POD-2G", facts=(_int(RANK, 10),))
    cohort = _cohort(
        scope,
        MethodPresentationCandidate(
            PreconditionerName("g"), DatasetQualifiedMethodPresentation(pod, dataset)
        ),
        MethodPresentationCandidate(
            PreconditionerName("s"), DatasetQualifiedMethodPresentation(pod, other)
        ),
    )
    assert len(prepare_presentations(cohort).methods) == 2


# --- realization ------------------------------------------------------------------


def _system(index: int, *facts: ScientificFact) -> SystemRealization:
    return SystemRealization(SystemId(f"s{index}"), MatrixId(index), facts)


def test_reduction_keeps_equal_facts_and_drops_varying_ones() -> None:
    systems = (
        _system(0, _int(COARSE_DIMENSION, 8), _int(SAMPLE_COUNT, 5)),
        _system(1, _int(COARSE_DIMENSION, 9), _int(SAMPLE_COUNT, 5)),
    )
    population = frozenset({SystemId("s0"), SystemId("s1")})
    assert reduce_realized_facts(systems, population) == (_int(SAMPLE_COUNT, 5),)
    assert reduce_realized_facts((), population) == ()


def test_reduction_requires_a_realization_for_every_system_of_the_population() -> None:
    systems = (_system(0, _int(SAMPLE_COUNT, 5)),)
    population = frozenset({SystemId("s0"), SystemId("s1")})

    assert reduce_realized_facts(systems, population) == ()
    assert reduce_realized_facts(systems, frozenset({SystemId("s0")})) == (_int(SAMPLE_COUNT, 5),)


def test_executed_method_allows_no_systems_but_requires_unique_system_ids(
    scope: ComparisonScopePresentation,
) -> None:
    prepared = prepare_presentations(_cohort(scope, _candidate("a", _asa()))).methods[0]

    assert ExecutedMethod(prepared, ()).systems == ()
    assert ExecutedMethod(prepared, (_system(0), _system(1))).prepared is prepared
    with pytest.raises(ValueError, match="unique"):
        ExecutedMethod(prepared, (_system(0), _system(0)))


def test_system_realization_rejects_configured_facts() -> None:
    with pytest.raises(ValueError, match="not a realized fact"):
        _system(0, _int(RANK, 3))
    assert _system(0, _int(COARSE_DIMENSION, 4))


def test_dataset_and_title_render_the_source_then_each_strategy(
    dataset: DatasetPresentation,
) -> None:
    from neuralls.domain.presentation.render import render_dataset, render_title

    policy = NotationPolicy.default()
    assert render_dataset(dataset, LEGEND_PROFILE, policy).text == (
        "Spectral 3D field | Gaussian CG Residuals (k=20, N=5000)"
    )
    assert (
        render_title(ComparisonScopePresentation(dataset), LEGEND_PROFILE, policy).text
        == "Spectral 3D field | Gaussian CG Residuals (k=20, N=5000)"
    )
    assert (
        render_title(EditorialPresentation("Spectral comparison"), LEGEND_PROFILE, policy).text
        == "Spectral comparison"
    )
