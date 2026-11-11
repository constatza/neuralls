"""Contract tests for the controlled scientific-fact catalog."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib import import_module
from types import ModuleType
from typing import Any

import pytest

from neuralls.shared.scientific_description.catalog import (
    FactVisibility,
    is_compact_by_policy,
    make_fact,
)
from neuralls.shared.scientific_description.models import (
    FieldKey,
    IntegerFact,
    ScientificDescription,
    TextFact,
    ValidatedLabel,
)
from neuralls.shared.scientific_description.policy import (
    LEGEND_PROFILE,
    NotationPolicy,
    RenderTarget,
)
from neuralls.shared.scientific_description.render import render, render_facts


@pytest.fixture
def catalog() -> ModuleType:
    """Load the catalog inside the test so a missing module is a focused RED failure."""
    try:
        return import_module("neuralls.shared.scientific_description.catalog")
    except ModuleNotFoundError:
        pytest.fail("the controlled scientific-fact catalog does not exist")


def test_catalog_orders_are_unique(catalog: ModuleType) -> None:
    """Two controlled facts must not silently occupy the same family-order position."""
    orders = [definition.order for definition in catalog.FACT_DEFINITIONS]
    assert len(orders) == len(set(orders))


def test_catalog_contains_required_compact_plain_symbols(catalog: ModuleType) -> None:
    """The catalog owns the compact notation required by publication labels."""
    symbols = {definition.plain_symbol for definition in catalog.FACT_DEFINITIONS}
    assert {
        "L",
        "k",
        "nᵢₜ",
        "cₘₐₓ",
        "ν₁",
        "ν₂",
        "ωₛ",
        "θ",
        "kᵣ",
        "nᵦ",
        "θₐd",
        "q",
        "γ",
        "r",
        "N",
        "Nₛ",
        "μ",
        "σ",
        "a",
        "b",
        "τ",
    } <= symbols


def test_fact_roles_are_closed_and_complete(catalog: ModuleType) -> None:
    """Role policy cannot drift into free-form strings."""
    assert {role.value for role in catalog.FactRole} == {
        "algorithmic",
        "reproducibility",
        "scheduling",
        "operational",
        "identity",
        "provenance",
    }


def test_fact_definitions_are_frozen_and_slotted(catalog: ModuleType) -> None:
    """Catalog invariants cannot be mutated after import."""
    definition = catalog.FACT_DEFINITIONS[0]
    assert not hasattr(definition, "__dict__")
    with pytest.raises(FrozenInstanceError):
        definition.order = 999


def _expected_value(definition: Any, *, plain: bool) -> str:
    """The spelling of the sample value used by the rendering tests, from the catalog alone."""
    if definition.categorical_formatter is not None:
        first = definition.categorical_formatter.values[0]
        return first.plain if plain else first.mathtext
    if definition.precision is None:
        return "2"
    return f"{1.25:.{definition.precision}g}"


def test_every_controlled_fact_renders_without_its_python_field_name(
    catalog: ModuleType,
) -> None:
    """Deleting a catalog symbol must not expose a snake-case identifier in plain output."""
    for definition in catalog.FACT_DEFINITIONS:
        if definition.visibility_policy.value == "never":
            continue
        if definition.key == catalog.MODEL_ARCHITECTURE:
            continue
        if definition.categorical_formatter is not None:
            fact = make_fact(definition.key, definition.categorical_formatter.values[0].raw)
        elif definition.precision is not None:
            fact = make_fact(definition.key, 1.25)
        else:
            fact = make_fact(definition.key, 2)

        rendered = render_facts((fact,), NotationPolicy.default(RenderTarget.PLAIN))
        assert rendered == f"{definition.plain_symbol}={_expected_value(definition, plain=True)}"


def test_mathtext_uses_catalog_markup_not_raw_identifiers(catalog: ModuleType) -> None:
    """MathText underscores come from controlled notation, never Python identifiers."""
    for definition in catalog.FACT_DEFINITIONS:
        if definition.visibility_policy.value == "never":
            continue
        if definition.key == catalog.MODEL_ARCHITECTURE:
            continue
        if definition.categorical_formatter is not None:
            fact = make_fact(definition.key, definition.categorical_formatter.values[0].raw)
        elif definition.precision is not None:
            fact = make_fact(definition.key, 1.25)
        else:
            fact = make_fact(definition.key, 2)

        rendered = render_facts((fact,), NotationPolicy.default(RenderTarget.MATHTEXT))
        expected_value = _expected_value(definition, plain=False)
        assert rendered == f"${definition.mathtext_symbol}={expected_value}$"


def test_unknown_fact_key_is_rejected(catalog: ModuleType) -> None:
    """Removing raw-key fallback must turn an unregistered key into an explicit error."""
    key = FieldKey("unregistered_scientific_parameter")
    fact = IntegerFact(key, 3)

    with pytest.raises(catalog.UnknownScientificFactError):
        catalog.fact_definition(key)
    with pytest.raises(catalog.UnknownScientificFactError):
        render_facts((fact,), NotationPolicy.default())


def test_unknown_categorical_value_is_rejected(catalog: ModuleType) -> None:
    """A category typo built around the factory must not leak its raw spelling into presentation."""
    fact = TextFact(catalog.WEIGHTING, "unknown_weighting")

    with pytest.raises(catalog.UnknownCategoricalValueError):
        render_facts((fact,), NotationPolicy.default())


def test_identity_only_fact_is_excluded_by_both_rendering_entry_points(
    catalog: ModuleType,
) -> None:
    """A NEVER-visible machine identity cannot leak through either public renderer."""
    fact = make_fact(catalog.ASSIGNMENT, 123)
    description = ScientificDescription(base_name="Neural", facts=(fact,))

    assert render(description, LEGEND_PROFILE, NotationPolicy.default()).text == "Neural"
    assert render_facts((fact,), NotationPolicy.default()) == ""


def test_uncontrolled_model_architecture_text_is_rejected_by_both_rendering_entry_points(
    catalog: ModuleType,
) -> None:
    """An arbitrary Python class name is not validated scientific presentation text."""
    fact = TextFact(catalog.MODEL_ARCHITECTURE, "unknown_python_class")
    description = ScientificDescription(base_name="Neural", facts=(fact,))

    with pytest.raises(ValueError, match="controlled formatter"):
        render(description, LEGEND_PROFILE, NotationPolicy.default())
    with pytest.raises(ValueError, match="controlled formatter"):
        render_facts((fact,), NotationPolicy.default())


@pytest.fixture
def full_bamg_description(catalog: ModuleType) -> ScientificDescription:
    """All configured BAMG algorithm facts from the approved display contract."""
    return ScientificDescription(
        base_name="BAMG",
        facts=(
            make_fact(catalog.MULTIGRID_LEVELS, 10),
            make_fact(catalog.RELAXATION_TEST_VECTORS, 8),
            make_fact(catalog.RELAXATION_SWEEPS, 4),
            make_fact(catalog.BOOTSTRAP_CYCLES, 2),
            make_fact(catalog.COMPATIBLE_RELAXATION_SWEEPS, 5),
            make_fact(catalog.COMPATIBLE_RELAXATION_TOLERANCE, 0.7),
            make_fact(catalog.ALGEBRAIC_DISTANCE_THRESHOLD, 0.5),
            make_fact(catalog.CALIBER, 4),
            make_fact(catalog.CALIBER_GROWTH_PENALTY, 1.5),
            make_fact(catalog.USE_LSR, "on"),
            make_fact(catalog.MAX_COARSE_DIMENSION, 10),
        ),
    )


def test_legacy_length_budget_keeps_every_bamg_fact_and_returns_layout_hints(
    catalog: ModuleType, full_bamg_description: ScientificDescription
) -> None:
    """Restoring the old drop loop must fail by losing an algorithmic BAMG distinction."""
    rendered = render(full_bamg_description, LEGEND_PROFILE, NotationPolicy.default())

    assert rendered.text == (
        "BAMG (L=10, kᵣ=8, η=4, nᵦ=2, ν=5, δ=0.7, θₐd=0.5, q=4, γ=1.5, LSR=on, cₘₐₓ=10)"
    )
    assert len(rendered.preferred_wrap_boundaries) == len(full_bamg_description.facts) - 1
    assert all(
        rendered.text[index - 2 : index] == ", " for index in rendered.preferred_wrap_boundaries
    )


def test_display_text_fact_renders_validated_label_in_every_target(catalog: ModuleType) -> None:
    """A validated human-authored name renders without escaping or identifier text."""
    fact = make_fact(catalog.MODEL_ARCHITECTURE, ValidatedLabel("Graph Net"))

    assert render_facts((fact,), NotationPolicy.default(RenderTarget.PLAIN)) == "arch=Graph Net"
    assert (
        render_facts((fact,), NotationPolicy.default(RenderTarget.MATHTEXT))
        == "$\\mathrm{arch}=\\mathrm{Graph\\ Net}$"
    )


@pytest.mark.parametrize(
    "text", ["", " lead", "trail ", "snake_case", "Two  spaces", "Net$", "a\nb"]
)
def test_validated_label_rejects_non_human_text(text: str) -> None:
    """Identifiers, markup characters, and control characters cannot become labels."""
    with pytest.raises(ValueError):
        ValidatedLabel(text)


@pytest.mark.parametrize(
    ("visibility", "expected"),
    [
        (FactVisibility.ALWAYS, True),
        (FactVisibility.WHEN_NON_DEFAULT, True),
        (FactVisibility.WHEN_DISTINGUISHING, False),
        (FactVisibility.DETAILED_ONLY, False),
        (FactVisibility.NEVER, False),
    ],
)
def test_compact_by_policy_covers_every_visibility(
    visibility: FactVisibility, expected: bool
) -> None:
    assert is_compact_by_policy(visibility) is expected
