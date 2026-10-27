"""The facts codec round-trips typed values and validates on decode."""

from __future__ import annotations

import json

import pytest

from neuralls.domain.presentation.provenance import (
    DatasetGenerationProvenance,
    DatasetPresentation,
    DatasetSourceProvenance,
    DerivedSeed,
    ExplicitSeed,
    NoSeed,
    StrategyProvenance,
)
from neuralls.platform.storage.facts_codec import (
    decode_dataset_presentation,
    decode_facts,
    decode_recipe,
    encode_dataset_presentation,
    encode_facts,
    encode_recipe,
)
from neuralls.shared.rhs_recipe import (
    DatasetRecipe,
    GaussianRecipe,
    RandomlyMasked,
    RawLhsRecipe,
    RawRhsRecipe,
    RawSourceIdentity,
    RhsRecipe,
    SinusoidalRecipe,
)
from neuralls.shared.scientific_description.catalog import (
    BETA,
    MODEL_ARCHITECTURE,
    PROBE_DISTRIBUTION,
    make_fact,
)
from neuralls.shared.scientific_description.models import (
    ScientificDescription,
    ValidatedLabel,
)
from neuralls.shared.types import GenerationStrategyKind


@pytest.fixture
def dataset() -> DatasetPresentation:
    """A two-strategy dataset exercising every fact kind."""
    return DatasetPresentation(
        source=DatasetSourceProvenance(
            ScientificDescription(
                "Spectral 3D field",
                facts=(make_fact(MODEL_ARCHITECTURE, ValidatedLabel("Net")),),
            )
        ),
        generation=DatasetGenerationProvenance(
            strategies=(
                StrategyProvenance(
                    GenerationStrategyKind.SMOOTHER_FILTERED_PROBES,
                    rows=5000,
                    parameters=(make_fact(PROBE_DISTRIBUTION, "gaussian"),),
                    seed=ExplicitSeed(7),
                ),
                StrategyProvenance(
                    GenerationStrategyKind.NORMAL,
                    rows=10,
                    parameters=(make_fact(BETA, 0.125),),
                    seed=DerivedSeed(),
                ),
                StrategyProvenance(
                    GenerationStrategyKind.KRYLOV, rows=3, parameters=(), seed=NoSeed()
                ),
            ),
            generated_rows=5013,
            normalization="rhs",
            root_seed=99,
        ),
    )


def test_dataset_presentation_round_trips_exactly(dataset: DatasetPresentation) -> None:
    assert decode_dataset_presentation(encode_dataset_presentation(dataset)) == dataset


def test_unregistered_fact_key_is_rejected_on_decode() -> None:
    with pytest.raises(KeyError):
        decode_facts([{"key": "mystery", "kind": "integer", "value": 1}])


def test_label_values_are_revalidated_on_decode() -> None:
    payload = encode_facts((make_fact(MODEL_ARCHITECTURE, ValidatedLabel("Net")),))
    payload[0]["value"] = "bad_name"
    with pytest.raises(ValueError):
        decode_facts(payload)


def test_unknown_normalization_is_rejected_on_decode(dataset: DatasetPresentation) -> None:
    payload = encode_dataset_presentation(dataset)
    payload["generation"]["normalization"] = "l2"
    with pytest.raises(ValueError, match="normalization"):
        decode_dataset_presentation(payload)


def test_rows_and_seeds_are_stored_once_and_the_count_is_not_a_fact(
    dataset: DatasetPresentation,
) -> None:
    payload = encode_dataset_presentation(dataset)
    strategies = payload["generation"]["strategies"]

    assert [strategy["rows"] for strategy in strategies] == [5000, 10, 3]
    assert [strategy["seed"]["kind"] for strategy in strategies] == ["explicit", "derived", "none"]
    assert strategies[0]["seed"]["value"] == 7
    assert payload["generation"]["root_seed"] == 99
    assert [strategy["parameters"] for strategy in strategies] == [
        [{"key": "probe_distribution", "kind": "text", "value": "gaussian"}],
        [{"key": "beta", "kind": "decimal", "value": 0.125}],
        [],
    ]


def test_decoding_rejects_rows_that_disagree_with_the_dataset_total(
    dataset: DatasetPresentation,
) -> None:
    """Kills: trusting a stored total over the per-strategy rows."""
    payload = encode_dataset_presentation(dataset)
    payload["generation"]["strategies"][0]["rows"] = 4999
    with pytest.raises(ValueError, match="sum to"):
        decode_dataset_presentation(payload)


def test_unknown_seed_kind_is_rejected_on_decode(dataset: DatasetPresentation) -> None:
    payload = encode_dataset_presentation(dataset)
    payload["generation"]["strategies"][0]["seed"] = {"kind": "random"}
    with pytest.raises(ValueError, match="seed kind"):
        decode_dataset_presentation(payload)


@pytest.mark.parametrize(
    "recipe",
    [
        GaussianRecipe(-1.5, 2e-05, RandomlyMasked(0.1)),
        SinusoidalRecipe(6.283185307179586),
        RawLhsRecipe(RawSourceIdentity(3, "0123456789abcdef")),
        RawRhsRecipe(RawSourceIdentity(0, "ff"), RandomlyMasked(0.5)),
        DatasetRecipe(),
    ],
)
def test_recipe_round_trips_exactly_through_json(recipe: RhsRecipe) -> None:
    assert decode_recipe(json.loads(json.dumps(encode_recipe(recipe)))) == recipe


def test_unknown_recipe_kind_is_rejected_on_decode() -> None:
    with pytest.raises(ValueError, match="Unknown recipe kind 'uniform'"):
        decode_recipe({"kind": "uniform", "mask": {"kind": "none"}})


def test_unknown_recipe_mask_is_rejected_on_decode() -> None:
    with pytest.raises(ValueError, match="Unknown recipe mask kind 'banded'"):
        decode_recipe({"kind": "dataset", "mask": {"kind": "banded"}})
