"""Facts are checked when they are built: key, value kind, category, and uniqueness."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import neuralls
from neuralls.domain.presentation.strategy_names import strategy_base_name
from neuralls.shared.scientific_description import catalog
from neuralls.shared.scientific_description.catalog import (
    ASSIGNMENT,
    FACT_DEFINITIONS,
    ITERATION_COUNT,
    MODEL_ARCHITECTURE,
    RANK,
    SHARED_SYMBOL_GROUPS,
    THRESHOLD,
    WEIGHTING,
    CatalogError,
    FactDefinition,
    FactValueKindError,
    UnknownCategoricalValueError,
    UnknownScientificFactError,
    make_fact,
    validate_catalog,
)
from neuralls.shared.scientific_description.models import (
    DecimalFact,
    DisplayTextFact,
    DuplicateFactKeyError,
    FactValueKind,
    FieldKey,
    IntegerFact,
    ScientificDescription,
    TextFact,
    ValidatedLabel,
    fact_identity,
    require_unique_keys,
)
from neuralls.shared.types import GenerationStrategyKind, PreconditionerFamily


def _sample(definition: FactDefinition, kind: FactValueKind) -> int | float | str | ValidatedLabel:
    """One valid value of ``kind`` for ``definition``."""
    match kind:
        case FactValueKind.INTEGER:
            return 3
        case FactValueKind.DECIMAL:
            return 0.5
        case FactValueKind.TEXT:
            match definition.categorical_formatter:
                case None:
                    return "anything"
                case formatter:
                    return formatter.values[0].raw
        case FactValueKind.LABEL:
            return ValidatedLabel("Graph Net")


# --- the factory --------------------------------------------------------------------


@pytest.mark.parametrize("definition", FACT_DEFINITIONS, ids=lambda d: d.key.value)
def test_factory_builds_every_kind_each_definition_allows(definition: FactDefinition) -> None:
    for kind in definition.value_kinds:
        fact = make_fact(definition.key, _sample(definition, kind))
        assert fact_identity(fact).kind is kind


@pytest.mark.parametrize("definition", FACT_DEFINITIONS, ids=lambda d: d.key.value)
def test_factory_rejects_every_kind_a_definition_does_not_allow(definition: FactDefinition) -> None:
    for kind in set(FactValueKind) - definition.value_kinds:
        with pytest.raises(FactValueKindError):
            make_fact(definition.key, _sample(definition, kind))


def test_factory_rejects_an_unregistered_key() -> None:
    with pytest.raises(UnknownScientificFactError):
        make_fact(FieldKey("not_a_catalog_key"), 1)


def test_factory_rejects_a_bogus_category() -> None:
    with pytest.raises(UnknownCategoricalValueError):
        make_fact(WEIGHTING, "bogus")


def test_factory_rejects_text_for_a_numeric_key() -> None:
    with pytest.raises(FactValueKindError, match="integer"):
        make_fact(ITERATION_COUNT, "15")


def test_factory_rejects_an_integer_for_a_decimal_only_key() -> None:
    with pytest.raises(FactValueKindError, match="decimal"):
        make_fact(THRESHOLD, 1)


def test_factory_rejects_a_label_for_a_non_label_key() -> None:
    with pytest.raises(FactValueKindError):
        make_fact(ITERATION_COUNT, ValidatedLabel("Net"))


def test_factory_rejects_unvalidated_text_for_a_label_key() -> None:
    with pytest.raises(FactValueKindError):
        make_fact(MODEL_ARCHITECTURE, "SomePythonClass")


@pytest.mark.parametrize("key", [ITERATION_COUNT, THRESHOLD, WEIGHTING, ASSIGNMENT])
def test_factory_rejects_bool(key: FieldKey) -> None:
    with pytest.raises(TypeError, match="bool"):
        make_fact(key, True)


def test_factory_rejects_a_value_type_no_fact_has() -> None:
    with pytest.raises(TypeError):
        make_fact(ITERATION_COUNT, [1])


def test_integer_and_decimal_constructors_reject_bool() -> None:
    with pytest.raises(TypeError):
        IntegerFact(ITERATION_COUNT, True)
    with pytest.raises(TypeError):
        DecimalFact(THRESHOLD, True)


def test_rank_takes_a_count_or_an_energy_fraction() -> None:
    assert isinstance(make_fact(RANK, 10), IntegerFact)
    assert isinstance(make_fact(RANK, 0.99), DecimalFact)


# --- unique keys ----------------------------------------------------------------------


def test_require_unique_keys_names_the_repeated_key_and_owner() -> None:
    facts = (make_fact(ITERATION_COUNT, 1), make_fact(ITERATION_COUNT, 2))
    with pytest.raises(DuplicateFactKeyError, match="iteration_count") as error:
        require_unique_keys(facts, "SomeCarrier")
    assert str(error.value) == "SomeCarrier holds fact 'iteration_count' more than once"


def test_description_rejects_a_key_repeated_between_facts_and_context_facts() -> None:
    with pytest.raises(DuplicateFactKeyError):
        ScientificDescription(
            "AMG",
            facts=(make_fact(ITERATION_COUNT, 1),),
            context_facts=(make_fact(ITERATION_COUNT, 2),),
        )


def test_description_rejects_a_key_repeated_within_facts() -> None:
    fact = make_fact(ITERATION_COUNT, 1)
    with pytest.raises(DuplicateFactKeyError):
        ScientificDescription("AMG", facts=(fact, fact))


# --- typed identity ---------------------------------------------------------------------


def test_fact_identity_separates_one_float_and_the_integer_one() -> None:
    integer = fact_identity(IntegerFact(RANK, 1))
    decimal = fact_identity(DecimalFact(RANK, 1.0))
    text = fact_identity(TextFact(WEIGHTING, "1"))
    assert len({integer, decimal, text}) == 3
    assert integer != decimal
    assert integer == fact_identity(IntegerFact(RANK, 1))


def test_fact_identity_keeps_a_label_apart_from_text_of_the_same_spelling() -> None:
    label = fact_identity(DisplayTextFact(MODEL_ARCHITECTURE, ValidatedLabel("raw")))
    text = fact_identity(TextFact(MODEL_ARCHITECTURE, "raw"))
    assert label != text


# --- base names -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["IC0", "AMG", "POD-2G", "aSA", "BAMG", "Neural", "Neural AMG"])
def test_method_base_names_are_accepted(name: str) -> None:
    assert ScientificDescription(name).base_name == name


@pytest.mark.parametrize("name", ["", "train_job_17", "my_job", "Net ", " Net", "a  b", "AMG(2)"])
def test_identifier_shaped_base_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        ScientificDescription(name)


def test_every_producer_base_name_is_a_valid_label() -> None:
    producers = [family.abbreviation() for family in PreconditionerFamily]
    producers += [strategy_base_name(kind) for kind in GenerationStrategyKind]
    for name in producers:
        ScientificDescription(name)


# --- catalog self-validation -------------------------------------------------------------


def _clone(definition: FactDefinition, **changes: object) -> FactDefinition:
    return replace(definition, **changes)


def test_the_shipped_catalog_validates() -> None:
    assert len(validate_catalog(FACT_DEFINITIONS, SHARED_SYMBOL_GROUPS)) == len(FACT_DEFINITIONS)


def test_catalog_rejects_a_duplicate_key() -> None:
    first = FACT_DEFINITIONS[0]
    with pytest.raises(CatalogError, match="duplicate fact key"):
        validate_catalog((first, first), frozenset())


def test_catalog_rejects_a_duplicate_order() -> None:
    first, second = FACT_DEFINITIONS[0], FACT_DEFINITIONS[1]
    clash = _clone(second, order=first.order)
    with pytest.raises(CatalogError, match="share order"):
        validate_catalog((first, clash), frozenset())


def test_catalog_rejects_an_undeclared_shared_symbol() -> None:
    first, second = FACT_DEFINITIONS[0], FACT_DEFINITIONS[1]
    clash = _clone(second, plain_symbol=first.plain_symbol)
    with pytest.raises(CatalogError, match="without being declared"):
        validate_catalog((first, clash), frozenset())


def test_catalog_accepts_a_declared_shared_symbol() -> None:
    first, second = FACT_DEFINITIONS[0], FACT_DEFINITIONS[1]
    clash = _clone(second, plain_symbol=first.plain_symbol)
    declared = frozenset({frozenset({first.key, clash.key})})
    assert validate_catalog((first, clash), declared)


def test_catalog_rejects_a_declared_shared_symbol_that_no_longer_collides() -> None:
    first, second = FACT_DEFINITIONS[0], FACT_DEFINITIONS[1]
    declared = frozenset({frozenset({first.key, second.key})})
    with pytest.raises(CatalogError, match="no longer collides"):
        validate_catalog((first, second), declared)


def test_definition_rejects_categories_without_a_text_kind() -> None:
    weighting = next(d for d in FACT_DEFINITIONS if d.key == WEIGHTING)
    with pytest.raises(ValueError, match="categories"):
        _clone(weighting, value_kinds=frozenset({FactValueKind.INTEGER}))


def test_snapshot_key_names_what_it_counts() -> None:
    assert catalog.SNAPSHOT_COUNT.value == "snapshot_count"


# --- the factory is the only production constructor ---------------------------------------

_FACT_CONSTRUCTORS = frozenset({"IntegerFact", "DecimalFact", "TextFact", "DisplayTextFact"})


def test_production_code_builds_facts_only_through_the_catalog_factory() -> None:
    """The render-time category and kind checks stay a backstop, never a reachable path."""
    package = Path(neuralls.__file__).parent
    allowed = {package / "shared" / "scientific_description" / "catalog.py"}
    offenders: list[str] = []
    for source in package.rglob("*.py"):
        if source in allowed:
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _FACT_CONSTRUCTORS
            ):
                offenders.append(f"{source.relative_to(package)}:{node.lineno}")
    assert offenders == []
