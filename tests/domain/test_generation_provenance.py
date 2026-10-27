"""Every generation strategy is described by all of its algorithmic parameters."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neuralls.domain.generation.descriptions import FIELD_ROLES, describe_strategy
from neuralls.domain.generation.provenance import describe_dataset_generation
from neuralls.domain.generation.runner import registered_config_type
from neuralls.domain.generation.sample_writer import StrategyRowCount
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec
from neuralls.domain.generation.strategy_configs import (
    BaseStrategyConfig,
    KrylovConfig,
    SmootherFilteredProbesConfig,
)
from neuralls.shared.scientific_description import (
    LEGEND_PROFILE,
    NotationPolicy,
    ScientificDescription,
    render,
)
from neuralls.shared.scientific_description.catalog import FactRole
from neuralls.shared.scientific_description.presentation import (
    DatasetGenerationProvenance,
    DatasetPresentation,
    DatasetSourceProvenance,
    DerivedSeed,
    ExplicitSeed,
    NoSeed,
    dataset_qualifier,
    strategy_base_name,
)
from neuralls.shared.types import GenerationStrategyKind

STRATEGY_NAMES = [kind.value for kind in GenerationStrategyKind]


def _plain(kind: GenerationStrategyKind, cfg: BaseStrategyConfig, rows: int) -> str:
    strategy = describe_strategy(kind, cfg, rows=rows, seed=ExplicitSeed(cfg.seed or 0))
    description = ScientificDescription(strategy_base_name(kind), strategy.facts)
    return render(description, LEGEND_PROFILE, NotationPolicy.default()).text


def _dataset(generation: DatasetGenerationProvenance) -> DatasetPresentation:
    return DatasetPresentation(
        DatasetSourceProvenance(ScientificDescription("Spectral 3D field")), generation
    )


def test_krylov_label_states_the_subspace_dimension_and_seed() -> None:
    cfg = KrylovConfig(samples=100, krylov_iters=12, seed=7)
    assert _plain(GenerationStrategyKind.KRYLOV, cfg, 100) == (
        "Krylov Subspace Solutions (m=12, N=100, s=7)"
    )


def test_krylov_variants_with_different_dimensions_render_differently() -> None:
    low = KrylovConfig(samples=100, krylov_iters=5, seed=7)
    high = KrylovConfig(samples=100, krylov_iters=9, seed=7)
    assert _plain(GenerationStrategyKind.KRYLOV, low, 100) != _plain(
        GenerationStrategyKind.KRYLOV, high, 100
    )


def test_smoother_probe_label_orders_sweeps_count_then_distribution() -> None:
    cfg = SmootherFilteredProbesConfig(samples=5000, stop=10, probe_distribution="gaussian", seed=1)
    assert _plain(GenerationStrategyKind.SMOOTHER_FILTERED_PROBES, cfg, 5000) == (
        "Jacobi-Filtered Probes (ν=10, N=5000, p=𝒩, ω=auto, s=1)"
    )


def test_seed_is_provenance_but_not_part_of_the_dataset_qualifier() -> None:
    spec = DatasetSpec(
        mixture=MixtureSpec(
            counts={"krylov": 10}, seed=3, strategy_overrides={"krylov": {"seed": 9}}
        )
    )
    generation = describe_dataset_generation(spec, (StrategyRowCount("krylov", 10),))
    assert generation.strategies[0].seed == ExplicitSeed(9)
    assert "seed" in [fact.key.value for fact in generation.strategies[0].facts]

    qualifier_keys = [fact.key.value for fact in dataset_qualifier(_dataset(generation))[0].facts]
    assert "seed" not in qualifier_keys


def test_unseeded_strategies_derive_from_the_mixture_seed() -> None:
    spec = DatasetSpec(mixture=MixtureSpec(counts={"krylov": 10}, seed=3))
    generation = describe_dataset_generation(spec, (StrategyRowCount("krylov", 10),))
    assert generation.root_seed == 3
    assert generation.strategies[0].seed == DerivedSeed()
    assert "seed" not in [fact.key.value for fact in generation.strategies[0].facts]


def test_a_strategy_explicitly_unseeded_is_recorded_as_such() -> None:
    spec = DatasetSpec(
        mixture=MixtureSpec(counts={"krylov": 4}, strategy_overrides={"krylov": {"seed": None}})
    )
    generation = describe_dataset_generation(spec, (StrategyRowCount("krylov", 4),))
    assert generation.strategies[0].seed == NoSeed()


def test_counts_come_from_the_realized_rows_not_the_spec_budgets() -> None:
    """Kills: recomputing rows from ``mix``/``total`` or ``counts`` instead of the writer."""
    spec = DatasetSpec(mixture=MixtureSpec(mix={"krylov": 1.0, "normal": 1.0}, total=20, seed=11))
    realized = (StrategyRowCount("krylov", 7), StrategyRowCount("normal", 13))
    generation = describe_dataset_generation(spec, realized)

    assert [s.kind for s in generation.strategies] == [
        GenerationStrategyKind.KRYLOV,
        GenerationStrategyKind.NORMAL,
    ]
    assert [s.rows for s in generation.strategies] == [7, 13]
    assert generation.generated_rows == 20
    assert generation.normalization == "matrix"


def test_open_ended_single_strategy_reports_realized_rows() -> None:
    spec = DatasetSpec(
        mixture=MixtureSpec(
            counts={"solution_archive": -1},
            strategy_overrides={"solution_archive": {"solutions_glob": "x/*.npy"}},
        )
    )
    generation = describe_dataset_generation(spec, (StrategyRowCount("solution_archive", 37),))
    facts = {f.key.value: f.value for f in generation.strategies[0].facts}
    assert facts["sample_count"] == 37


def test_archive_fed_by_a_solution_file_has_no_glob_and_still_describes() -> None:
    spec = DatasetSpec(mixture=MixtureSpec(counts={"solution_archive": 5}))
    generation = describe_dataset_generation(spec, (StrategyRowCount("solution_archive", 5),))
    assert generation.strategies[0].kind is GenerationStrategyKind.SOLUTION_ARCHIVE


def test_only_strategies_that_wrote_rows_are_part_of_the_provenance() -> None:
    spec = DatasetSpec(mixture=MixtureSpec(counts={"krylov": 4, "normal": 0}))
    generation = describe_dataset_generation(spec, (StrategyRowCount("krylov", 4),))
    assert [s.kind for s in generation.strategies] == [GenerationStrategyKind.KRYLOV]


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_every_registered_strategy_field_is_classified(name: str) -> None:
    """A new strategy-config field fails here until it is explicitly classified."""
    import neuralls.domain.generation.strategies  # noqa: F401  (registers every strategy)

    try:
        config_type = registered_config_type(name)
    except KeyError:
        pytest.skip(f"{name} is a canonical alias without its own registration")
    roles = FIELD_ROLES.get(config_type, FIELD_ROLES[BaseStrategyConfig])
    assert set(roles) == set(config_type.model_fields)


@pytest.mark.parametrize("model", list(FIELD_ROLES), ids=lambda model: model.__name__)
def test_field_roles_match_model_fields(model: type[BaseModel]) -> None:
    assert set(FIELD_ROLES[model]) == set(model.model_fields)


def test_paths_and_selection_are_never_algorithmic_facts() -> None:
    for roles in FIELD_ROLES.values():
        for name in ("solutions_glob", "rhs_glob", "file_indices", "skip"):
            if name in roles:
                assert roles[name] in {FactRole.PROVENANCE, FactRole.OPERATIONAL}
