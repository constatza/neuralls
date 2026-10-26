"""JSON encoding of scientific facts, descriptions, and dataset presentations.

Persists field keys and typed values only, never symbols, rendered text, or MathText,
so notation can change without invalidating stored data. Decoding re-applies every
model validation, so a malformed payload cannot become a presentation object.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from neuralls.shared.scientific_description import (
    DecimalFact,
    DisplayTextFact,
    FieldImportance,
    FieldKey,
    IntegerFact,
    ScientificDescription,
    ScientificFact,
    TextFact,
    ValidatedLabel,
)
from neuralls.shared.scientific_description.catalog import fact_definition
from neuralls.shared.scientific_description.presentation import (
    DatasetGenerationProvenance,
    DatasetPresentation,
    DatasetSourceProvenance,
    StrategyProvenance,
)
from neuralls.shared.types import GenerationStrategyKind, NormalizeType

__all__ = [
    "decode_dataset_presentation",
    "decode_facts",
    "encode_dataset_presentation",
    "encode_facts",
]

type Payload = Mapping[str, Any]

_NORMALIZATIONS: tuple[NormalizeType, ...] = ("none", "matrix", "rhs")


def _encode_fact(fact: ScientificFact) -> dict[str, Any]:
    match fact:
        case IntegerFact(value=value):
            kind, encoded = "integer", value
        case DecimalFact(value=value):
            kind, encoded = "decimal", value
        case TextFact(value=value):
            kind, encoded = "text", value
        case DisplayTextFact(value=label):
            kind, encoded = "label", label.text
    return {
        "key": fact.key.value,
        "kind": kind,
        "value": encoded,
        "importance": fact.importance.value,
    }


def _decode_fact(payload: Payload) -> ScientificFact:
    key = FieldKey(str(payload["key"]))
    fact_definition(key)  # an unregistered key is rejected, never carried along
    importance = FieldImportance(payload["importance"])
    value = payload["value"]
    match payload["kind"]:
        case "integer":
            return IntegerFact(key, int(value), importance)
        case "decimal":
            return DecimalFact(key, float(value), importance)
        case "text":
            return TextFact(key, str(value), importance)
        case "label":
            return DisplayTextFact(key, ValidatedLabel(str(value)), importance)
        case other:
            raise ValueError(f"Unknown fact kind {other!r}")


def encode_facts(facts: tuple[ScientificFact, ...]) -> list[dict[str, Any]]:
    """Encode facts as JSON-compatible dictionaries."""
    return [_encode_fact(fact) for fact in facts]


def decode_facts(payload: list[Payload]) -> tuple[ScientificFact, ...]:
    """Decode and validate facts produced by :func:`encode_facts`."""
    return tuple(_decode_fact(entry) for entry in payload)


def _encode_description(description: ScientificDescription) -> dict[str, Any]:
    return {
        "base_name": description.base_name,
        "facts": encode_facts(description.facts),
        "context_facts": encode_facts(description.context_facts),
    }


def _decode_description(payload: Payload) -> ScientificDescription:
    return ScientificDescription(
        base_name=str(payload["base_name"]),
        facts=decode_facts(payload["facts"]),
        context_facts=decode_facts(payload["context_facts"]),
    )


def encode_dataset_presentation(presentation: DatasetPresentation) -> dict[str, Any]:
    """Encode a dataset presentation as JSON-compatible data."""
    generation = presentation.generation
    return {
        "source": _encode_description(presentation.source.source),
        "generation": {
            "strategies": [
                {"kind": strategy.kind.value, "facts": encode_facts(strategy.facts)}
                for strategy in generation.strategies
            ],
            "generated_rows": generation.generated_rows,
            "normalization": generation.normalization,
        },
    }


def decode_dataset_presentation(payload: Payload) -> DatasetPresentation:
    """Decode and validate a dataset presentation produced by the encoder."""
    generation = payload["generation"]
    normalization = generation["normalization"]
    if normalization not in _NORMALIZATIONS:
        raise ValueError(f"Unknown dataset normalization {normalization!r}")
    return DatasetPresentation(
        source=DatasetSourceProvenance(_decode_description(payload["source"])),
        generation=DatasetGenerationProvenance(
            strategies=tuple(
                StrategyProvenance(
                    kind=GenerationStrategyKind(strategy["kind"]),
                    facts=decode_facts(strategy["facts"]),
                )
                for strategy in generation["strategies"]
            ),
            generated_rows=int(generation["generated_rows"]),
            normalization=cast(NormalizeType, normalization),
        ),
    )
