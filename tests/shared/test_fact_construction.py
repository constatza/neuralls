"""Facts are checked when they are built: key, value kind, category, and uniqueness."""

from __future__ import annotations

from dataclasses import replace

import pytest

from neuralls.domain.presentation.strategy_names import strategy_base_name
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
    CatalogErrorReason,
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
    ScientificFact,
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


@pytest.mark.parametrize(
    ("key", "value", "got", "allowed"),
    [
        (ITERATION_COUNT, "15", FactValueKind.TEXT, frozenset({FactValueKind.INTEGER})),
        (THRESHOLD, 1, FactValueKind.INTEGER, frozenset({FactValueKind.DECIMAL})),
    ],
    ids=["text-for-integer-key", "integer-for-decimal-key"],
)
def test_factory_reports_the_kind_it_got_and_the_kinds_allowed(
    key: FieldKey, value: object, got: FactValueKind, allowed: frozenset[FactValueKind]
) -> None:
    with pytest.raises(FactValueKindError) as error:
        make_fact(key, value)
    assert (error.value.key, error.value.got, error.value.allowed) == (key, got, allowed)


@pytest.mark.parametrize("key", [ITERATION_COUNT, THRESHOLD, WEIGHTING, ASSIGNMENT])
def test_factory_rejects_bool(key: FieldKey) -> None:
    with pytest.raises(TypeError, match="bool"):
        make_fact(key, True)


def test_factory_rejects_a_value_type_no_fact_has() -> None:
    with pytest.raises(TypeError):
        make_fact(ITERATION_COUNT, [1])


@pytest.mark.parametrize(
    ("constructor", "key"),
    [(IntegerFact, ITERATION_COUNT), (DecimalFact, THRESHOLD)],
    ids=["integer", "decimal"],
)
def test_numeric_fact_constructors_reject_bool(
    constructor: type[IntegerFact | DecimalFact], key: FieldKey
) -> None:
    with pytest.raises(TypeError):
        constructor(key, True)


def test_rank_takes_a_count_or_an_energy_fraction() -> None:
    assert make_fact(RANK, 10) == IntegerFact(RANK, 10)
    assert make_fact(RANK, 0.99) == DecimalFact(RANK, 0.99)


# --- unique keys ----------------------------------------------------------------------


def test_require_unique_keys_names_the_repeated_key_and_owner() -> None:
    facts = (make_fact(ITERATION_COUNT, 1), make_fact(ITERATION_COUNT, 2))
    with pytest.raises(DuplicateFactKeyError) as error:
        require_unique_keys(facts, "SomeCarrier")
    assert (error.value.key, error.value.owner) == (ITERATION_COUNT, "SomeCarrier")
    assert str(error.value) == "SomeCarrier holds fact 'iteration_count' more than once"


@pytest.mark.parametrize(
    ("facts", "context_facts"),
    [
        ((make_fact(ITERATION_COUNT, 1),), (make_fact(ITERATION_COUNT, 2),)),
        ((make_fact(ITERATION_COUNT, 1), make_fact(ITERATION_COUNT, 1)), ()),
    ],
    ids=["between-facts-and-context", "within-facts"],
)
def test_description_rejects_a_repeated_key(
    facts: tuple[ScientificFact, ...], context_facts: tuple[ScientificFact, ...]
) -> None:
    with pytest.raises(DuplicateFactKeyError) as error:
        ScientificDescription("AMG", facts=facts, context_facts=context_facts)
    assert error.value.key == ITERATION_COUNT


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


@pytest.mark.parametrize("name", ["IC0", "AMG", "POD", "aSA", "BAMG", "Neural", "Neural AMG"])
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
    with pytest.raises(CatalogError) as error:
        validate_catalog((first, first), frozenset())
    assert error.value.reason is CatalogErrorReason.DUPLICATE_KEY


def test_catalog_rejects_a_duplicate_order() -> None:
    first, second = FACT_DEFINITIONS[0], FACT_DEFINITIONS[1]
    clash = _clone(second, order=first.order)
    with pytest.raises(CatalogError) as error:
        validate_catalog((first, clash), frozenset())
    assert error.value.reason is CatalogErrorReason.SHARED_ORDER


def test_catalog_rejects_an_undeclared_shared_symbol() -> None:
    first, second = FACT_DEFINITIONS[0], FACT_DEFINITIONS[1]
    clash = _clone(second, plain_symbol=first.plain_symbol)
    with pytest.raises(CatalogError) as error:
        validate_catalog((first, clash), frozenset())
    assert error.value.reason is CatalogErrorReason.UNDECLARED_SHARED_SYMBOL


def test_catalog_accepts_a_declared_shared_symbol() -> None:
    first, second = FACT_DEFINITIONS[0], FACT_DEFINITIONS[1]
    clash = _clone(second, plain_symbol=first.plain_symbol)
    declared = frozenset({frozenset({first.key, clash.key})})
    assert validate_catalog((first, clash), declared)


def test_catalog_rejects_a_declared_shared_symbol_that_no_longer_collides() -> None:
    first, second = FACT_DEFINITIONS[0], FACT_DEFINITIONS[1]
    declared = frozenset({frozenset({first.key, second.key})})
    with pytest.raises(CatalogError) as error:
        validate_catalog((first, second), declared)
    assert error.value.reason is CatalogErrorReason.STALE_SHARED_SYMBOL


def test_definition_rejects_categories_without_a_text_kind() -> None:
    weighting = next(d for d in FACT_DEFINITIONS if d.key == WEIGHTING)
    with pytest.raises(ValueError, match="categories"):
        _clone(weighting, value_kinds=frozenset({FactValueKind.INTEGER}))
