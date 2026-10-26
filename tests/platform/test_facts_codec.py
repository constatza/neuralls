"""The facts codec round-trips typed values and validates on decode."""

from __future__ import annotations

import pytest

from neuralls.platform.storage.facts_codec import (
    decode_dataset_presentation,
    decode_facts,
    encode_dataset_presentation,
    encode_facts,
)
from neuralls.shared.scientific_description import (
    DecimalFact,
    DisplayTextFact,
    FieldImportance,
    IntegerFact,
    ScientificDescription,
    TextFact,
    ValidatedLabel,
)
from neuralls.shared.scientific_description.catalog import (
    BETA,
    MODEL_ARCHITECTURE,
    PROBE_DISTRIBUTION,
    SAMPLE_COUNT,
)
from neuralls.shared.scientific_description.presentation import (
    DatasetGenerationProvenance,
    DatasetPresentation,
    DatasetSourceProvenance,
    StrategyProvenance,
)
from neuralls.shared.types import GenerationStrategyKind

ESSENTIAL = FieldImportance.ESSENTIAL


@pytest.fixture
def dataset() -> DatasetPresentation:
    """A two-strategy dataset exercising every fact kind."""
    return DatasetPresentation(
        source=DatasetSourceProvenance(
            ScientificDescription(
                "Spectral 3D field",
                facts=(DisplayTextFact(MODEL_ARCHITECTURE, ValidatedLabel("Net"), ESSENTIAL),),
            )
        ),
        generation=DatasetGenerationProvenance(
            strategies=(
                StrategyProvenance(
                    GenerationStrategyKind.SMOOTHER_FILTERED_PROBES,
                    (
                        IntegerFact(SAMPLE_COUNT, 5000, ESSENTIAL),
                        TextFact(PROBE_DISTRIBUTION, "gaussian", ESSENTIAL),
                    ),
                ),
                StrategyProvenance(
                    GenerationStrategyKind.NORMAL, (DecimalFact(BETA, 0.125, ESSENTIAL),)
                ),
            ),
            generated_rows=5010,
            normalization="rhs",
        ),
    )


def test_dataset_presentation_round_trips_exactly(dataset: DatasetPresentation) -> None:
    assert decode_dataset_presentation(encode_dataset_presentation(dataset)) == dataset


def test_unregistered_fact_key_is_rejected_on_decode() -> None:
    with pytest.raises(KeyError):
        decode_facts([{"key": "mystery", "kind": "integer", "value": 1, "importance": "essential"}])


def test_label_values_are_revalidated_on_decode() -> None:
    payload = encode_facts((DisplayTextFact(MODEL_ARCHITECTURE, ValidatedLabel("Net"), ESSENTIAL),))
    payload[0]["value"] = "bad_name"
    with pytest.raises(ValueError):
        decode_facts(payload)


def test_unknown_normalization_is_rejected_on_decode(dataset: DatasetPresentation) -> None:
    payload = encode_dataset_presentation(dataset)
    payload["generation"]["normalization"] = "l2"
    with pytest.raises(ValueError, match="normalization"):
        decode_dataset_presentation(payload)
