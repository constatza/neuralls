"""JSON encoding of scientific facts, descriptions, and dataset presentations.

Persists field keys and typed values only, never symbols, rendered text, or MathText,
so notation can change without invalidating stored data. Decoding re-applies every
model validation, so a malformed payload cannot become a presentation object.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, assert_never, cast, get_args

from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    DatasetQualifiedMethodPresentation,
    PreparedMethodPresentation,
    StandaloneMethodPresentation,
)
from neuralls.domain.presentation.provenance import (
    DatasetGenerationProvenance,
    DatasetPresentation,
    DatasetSourceProvenance,
    DerivedSeed,
    ExplicitSeed,
    NoSeed,
    StrategyProvenance,
    StrategySeed,
)
from neuralls.domain.solver.models.figure_scope import (
    AllDatasetsScope,
    DatasetScope,
    FigureScope,
    MethodScope,
    RecipeScope,
)
from neuralls.shared.scientific_description.catalog import make_fact
from neuralls.shared.scientific_description.models import (
    DecimalFact,
    DisplayTextFact,
    FactValueKind,
    FieldKey,
    IntegerFact,
    ScientificDescription,
    ScientificFact,
    TextFact,
    ValidatedLabel,
)
from neuralls.shared.types import GenerationStrategyKind, NormalizeType

__all__ = [
    "decode_dataset_presentation",
    "decode_facts",
    "decode_figure_scope",
    "decode_method_presentation",
    "encode_dataset_presentation",
    "encode_facts",
    "encode_figure_scope",
    "encode_method_parameters",
    "encode_method_presentation",
]

type Payload = Mapping[str, Any]

_NORMALIZATIONS: tuple[str, ...] = get_args(NormalizeType)


def _encode_fact(fact: ScientificFact) -> dict[str, Any]:
    match fact:
        case IntegerFact(value=value):
            kind, encoded = FactValueKind.INTEGER, value
        case DecimalFact(value=value):
            kind, encoded = FactValueKind.DECIMAL, value
        case TextFact(value=value):
            kind, encoded = FactValueKind.TEXT, value
        case DisplayTextFact(value=label):
            kind, encoded = FactValueKind.LABEL, label.text
        case _:
            assert_never(fact)
    return {"key": fact.key.value, "kind": kind.value, "value": encoded}


def _decode_fact(payload: Payload) -> ScientificFact:
    """Rebuild a fact through the catalog, so key, kind, and category are re-checked."""
    key = FieldKey(str(payload["key"]))
    value = payload["value"]
    kind = FactValueKind(payload["kind"])
    match kind:
        case FactValueKind.INTEGER:
            return make_fact(key, int(value))
        case FactValueKind.DECIMAL:
            return make_fact(key, float(value))
        case FactValueKind.TEXT:
            return make_fact(key, str(value))
        case FactValueKind.LABEL:
            return make_fact(key, ValidatedLabel(str(value)))
        case _:
            assert_never(kind)


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


def _fact_text(fact: ScientificFact) -> str:
    """A fact's value as the string an MLflow parameter stores."""
    encoded = _encode_fact(fact)["value"]
    return str(encoded)


def encode_method_parameters(presentation: PreparedMethodPresentation) -> dict[str, str]:
    """Flat, stable parameter names and values for every configured fact of a method.

    Names are ``method.<field key>`` for the method's own facts and
    ``training_dataset.strategy_<n>.<field key>`` for the facts of each generation
    strategy of a dataset-trained method. They carry field keys and typed values only,
    never rendered text, so notation can change without renaming a parameter.
    """
    configured = presentation.configured
    parameters = {"method.base_name": configured.method.base_name}
    for fact in presentation.detailed_facts:
        parameters[f"method.{fact.key.value}"] = _fact_text(fact)
    match configured:
        case StandaloneMethodPresentation():
            pass
        case DatasetQualifiedMethodPresentation(dataset=dataset):
            parameters["training_dataset.source"] = dataset.source.source.base_name
            for number, strategy in enumerate(dataset.generation.strategies):
                prefix = f"training_dataset.strategy_{number}"
                parameters[f"{prefix}.kind"] = strategy.kind.value
                for fact in strategy.facts:
                    parameters[f"{prefix}.{fact.key.value}"] = _fact_text(fact)
        case _:
            assert_never(configured)
    return parameters


def _encode_seed(seed: StrategySeed) -> dict[str, Any]:
    match seed:
        case ExplicitSeed(value=value):
            return {"kind": "explicit", "value": value}
        case DerivedSeed():
            return {"kind": "derived"}
        case NoSeed():
            return {"kind": "none"}
        case _:
            assert_never(seed)


def _decode_seed(payload: Payload) -> StrategySeed:
    match payload["kind"]:
        case "explicit":
            return ExplicitSeed(int(payload["value"]))
        case "derived":
            return DerivedSeed()
        case "none":
            return NoSeed()
        case other:
            raise ValueError(f"Unknown strategy seed kind {other!r}")


def encode_dataset_presentation(presentation: DatasetPresentation) -> dict[str, Any]:
    """Encode a dataset presentation as JSON-compatible data."""
    generation = presentation.generation
    return {
        "source": _encode_description(presentation.source.source),
        "generation": {
            "strategies": [
                {
                    "kind": strategy.kind.value,
                    "rows": strategy.rows,
                    "parameters": encode_facts(strategy.parameters),
                    "seed": _encode_seed(strategy.seed),
                }
                for strategy in generation.strategies
            ],
            "generated_rows": generation.generated_rows,
            "normalization": generation.normalization,
            "root_seed": generation.root_seed,
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
                    rows=int(strategy["rows"]),
                    parameters=decode_facts(strategy["parameters"]),
                    seed=_decode_seed(strategy["seed"]),
                )
                for strategy in generation["strategies"]
            ),
            generated_rows=int(generation["generated_rows"]),
            normalization=cast(NormalizeType, normalization),
            root_seed=int(generation["root_seed"]),
        ),
    )


def encode_method_presentation(presentation: ConfiguredMethodPresentation) -> dict[str, Any]:
    """Encode a configured method presentation, including its training dataset if it has one."""
    match presentation:
        case StandaloneMethodPresentation(method=method):
            return {"kind": "standalone", "method": _encode_description(method)}
        case DatasetQualifiedMethodPresentation(method=method, dataset=dataset):
            return {
                "kind": "dataset_qualified",
                "method": _encode_description(method),
                "dataset": encode_dataset_presentation(dataset),
            }
        case _:
            assert_never(presentation)


def decode_method_presentation(payload: Payload) -> ConfiguredMethodPresentation:
    """Decode and validate a method presentation produced by the encoder."""
    method = _decode_description(payload["method"])
    match payload["kind"]:
        case "standalone":
            return StandaloneMethodPresentation(method)
        case "dataset_qualified":
            return DatasetQualifiedMethodPresentation(
                method, decode_dataset_presentation(payload["dataset"])
            )
        case other:
            raise ValueError(f"Unknown method presentation kind {other!r}")


def encode_figure_scope(scope: FigureScope) -> dict[str, Any]:
    """Encode what a figure is about as keys and typed values."""
    match scope:
        case RecipeScope(dataset=dataset, recipe=recipe):
            return {
                "kind": "recipe",
                "recipe": recipe,
                "dataset": encode_dataset_presentation(dataset),
            }
        case DatasetScope(dataset=dataset):
            return {"kind": "dataset", "dataset": encode_dataset_presentation(dataset)}
        case AllDatasetsScope():
            return {"kind": "all_datasets"}
        case MethodScope(method=method):
            return {"kind": "method", "method": encode_method_presentation(method)}
        case _:
            assert_never(scope)


def decode_figure_scope(payload: Payload) -> FigureScope:
    """Decode and validate a figure scope produced by the encoder."""
    match payload["kind"]:
        case "recipe":
            return RecipeScope(
                decode_dataset_presentation(payload["dataset"]), str(payload["recipe"])
            )
        case "dataset":
            return DatasetScope(decode_dataset_presentation(payload["dataset"]))
        case "all_datasets":
            return AllDatasetsScope()
        case "method":
            return MethodScope(decode_method_presentation(payload["method"]))
        case other:
            raise ValueError(f"Unknown figure scope kind {other!r}")
