"""Contract tests for the controlled scientific-fact catalog."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from importlib import import_module
from types import ModuleType

import pytest

from neuralls.shared.scientific_description.models import (
    DecimalFact,
    FieldImportance,
    FieldKey,
    IntegerFact,
    ScientificDescription,
    TextFact,
)
from neuralls.shared.scientific_description.policy import (
    LEGEND_PROFILE,
    NotationPolicy,
    RenderTarget,
)
from neuralls.shared.scientific_description.render import render, render_facts


@dataclass(frozen=True, slots=True)
class ExpectedFactDefinition:
    """Literal expected catalog row, independent of production builders."""

    key: str
    role: str
    phase: str
    order: int
    precision: int | None
    plain_symbol: str
    mathtext_symbol: str
    categories: tuple[tuple[str, str, str], ...] = ()
    visibility: str = "always"


@pytest.fixture
def catalog() -> ModuleType:
    """Load the catalog inside the test so a missing module is a focused RED failure."""
    try:
        return import_module("neuralls.shared.scientific_description.catalog")
    except ModuleNotFoundError:
        pytest.fail("the controlled scientific-fact catalog does not exist")


@pytest.fixture
def expected_definitions() -> tuple[ExpectedFactDefinition, ...]:
    """Every fact controlled by the initial presentation vocabulary."""
    return (
        ExpectedFactDefinition("sample_count", "algorithmic", "realized", 10, None, "N", "N"),
        ExpectedFactDefinition(
            "system_dimension",
            "provenance",
            "realized",
            20,
            None,
            "n",
            "n",
            visibility="detailed_only",
        ),
        ExpectedFactDefinition("iteration_count", "algorithmic", "configured", 30, None, "k", "k"),
        ExpectedFactDefinition("krylov_dimension", "algorithmic", "configured", 40, None, "m", "m"),
        ExpectedFactDefinition("coarse_dimension", "algorithmic", "realized", 50, None, "c", "c"),
        ExpectedFactDefinition(
            "target_coarse_dimension", "algorithmic", "configured", 60, None, "cₜ", "c_t"
        ),
        ExpectedFactDefinition(
            "max_coarse_dimension", "algorithmic", "configured", 70, None, "cₘₐₓ", "c_{\\max}"
        ),
        ExpectedFactDefinition("multigrid_levels", "algorithmic", "configured", 80, None, "L", "L"),
        ExpectedFactDefinition("threshold", "algorithmic", "configured", 90, 3, "θ", "\\theta"),
        ExpectedFactDefinition(
            "damping",
            "algorithmic",
            "configured",
            100,
            3,
            "ω",
            "\\omega",
            (("auto", "auto", "\\mathrm{auto}"),),
        ),
        ExpectedFactDefinition("mean", "algorithmic", "configured", 110, 3, "μ", "\\mu"),
        ExpectedFactDefinition(
            "standard_deviation", "algorithmic", "configured", 120, 3, "σ", "\\sigma"
        ),
        ExpectedFactDefinition("lower_bound", "algorithmic", "configured", 130, 3, "a", "a"),
        ExpectedFactDefinition("upper_bound", "algorithmic", "configured", 140, 3, "b", "b"),
        ExpectedFactDefinition("drop_tolerance", "algorithmic", "configured", 150, 1, "τ", "\\tau"),
        ExpectedFactDefinition(
            "pre_smoothing_steps", "algorithmic", "configured", 160, None, "ν₁", "\\nu_1"
        ),
        ExpectedFactDefinition(
            "post_smoothing_steps", "algorithmic", "configured", 170, None, "ν₂", "\\nu_2"
        ),
        ExpectedFactDefinition(
            "smoother_damping",
            "algorithmic",
            "configured",
            180,
            3,
            "ωₛ",
            "\\omega_s",
            (("auto", "auto", "\\mathrm{auto}"),),
        ),
        ExpectedFactDefinition("candidate_count", "algorithmic", "configured", 190, None, "k", "k"),
        ExpectedFactDefinition(
            "candidate_iterations",
            "algorithmic",
            "configured",
            200,
            None,
            "nᵢₜ",
            "n_{\\mathrm{it}}",
        ),
        ExpectedFactDefinition(
            "relaxation_test_vectors", "algorithmic", "configured", 210, None, "kᵣ", "k_r"
        ),
        ExpectedFactDefinition(
            "relaxation_sweeps", "algorithmic", "configured", 220, None, "η", "\\eta"
        ),
        ExpectedFactDefinition(
            "bootstrap_cycles", "algorithmic", "configured", 230, None, "nᵦ", "n_b"
        ),
        ExpectedFactDefinition(
            "compatible_relaxation_sweeps", "algorithmic", "configured", 240, None, "ν", "\\nu"
        ),
        ExpectedFactDefinition(
            "compatible_relaxation_tolerance", "algorithmic", "configured", 250, 3, "δ", "\\delta"
        ),
        ExpectedFactDefinition(
            "algebraic_distance_threshold",
            "algorithmic",
            "configured",
            260,
            3,
            "θₐd",
            "\\theta_{\\mathrm{ad}}",
        ),
        ExpectedFactDefinition("caliber", "algorithmic", "configured", 270, None, "q", "q"),
        ExpectedFactDefinition(
            "caliber_growth_penalty", "algorithmic", "configured", 280, 3, "γ", "\\gamma"
        ),
        ExpectedFactDefinition(
            "use_lsr",
            "algorithmic",
            "configured",
            290,
            None,
            "LSR",
            "\\mathrm{LSR}",
            (("off", "off", "\\mathrm{off}"), ("on", "on", "\\mathrm{on}")),
        ),
        ExpectedFactDefinition("rank", "algorithmic", "configured", 300, 4, "r", "r"),
        ExpectedFactDefinition("snapshots", "algorithmic", "configured", 310, None, "Nₛ", "N_s"),
        ExpectedFactDefinition(
            "weighting",
            "algorithmic",
            "configured",
            320,
            None,
            "w",
            "w",
            (
                ("power_norm", "PN", "\\mathrm{PN}"),
                ("raw", "raw", "\\mathrm{raw}"),
                ("smoother_persistence", "SP", "\\mathrm{SP}"),
                ("uniform", "uniform", "\\mathrm{uniform}"),
            ),
        ),
        ExpectedFactDefinition(
            "metric",
            "algorithmic",
            "configured",
            330,
            None,
            "p",
            "p",
            (("a", "A", "A"), ("l2", "ℓ₂", "\\ell_2")),
        ),
        ExpectedFactDefinition("beta", "algorithmic", "configured", 340, 3, "β", "\\beta"),
        ExpectedFactDefinition("steps", "algorithmic", "configured", 350, None, "ν", "\\nu"),
        ExpectedFactDefinition(
            "probe_distribution",
            "algorithmic",
            "configured",
            360,
            None,
            "p",
            "p",
            (("gaussian", "𝒩", "\\mathcal{N}"), ("rademacher", "Rad", "\\mathrm{Rad}")),
        ),
        ExpectedFactDefinition(
            "coarsening_strategy",
            "algorithmic",
            "configured",
            370,
            None,
            "C",
            "C",
            (
                ("aggregation", "SA", "\\mathrm{SA}"),
                ("neural_pod", "Neural-POD", "\\mathrm{Neural-POD}"),
                ("pod", "POD", "\\mathrm{POD}"),
                ("target_dim", "target-c", "\\mathrm{target-c}"),
            ),
        ),
        ExpectedFactDefinition(
            "model_architecture", "algorithmic", "realized", 380, None, "arch", "\\mathrm{arch}"
        ),
        ExpectedFactDefinition(
            "assignment",
            "identity",
            "configured",
            390,
            None,
            "id",
            "\\mathrm{id}",
            visibility="never",
        ),
        ExpectedFactDefinition(
            "seed",
            "reproducibility",
            "configured",
            400,
            None,
            "s",
            "s",
            visibility="when_distinguishing",
        ),
    )


def test_catalog_is_exhaustive_and_pins_every_definition(
    catalog: ModuleType, expected_definitions: tuple[ExpectedFactDefinition, ...]
) -> None:
    """A wrong/missing role, phase, order, symbol, precision, category, or visibility must fail."""
    actual = catalog.FACT_DEFINITIONS
    assert len(actual) == len(expected_definitions)

    for definition, expected in zip(actual, expected_definitions, strict=True):
        categories = (
            ()
            if definition.categorical_formatter is None
            else tuple(
                (value.raw, value.plain, value.mathtext)
                for value in definition.categorical_formatter.values
            )
        )
        assert (
            definition.key.value,
            definition.role.value,
            definition.phase.value,
            definition.order,
            definition.precision,
            definition.plain_symbol,
            definition.mathtext_symbol,
            categories,
            definition.visibility_policy.value,
        ) == (
            expected.key,
            expected.role,
            expected.phase,
            expected.order,
            expected.precision,
            expected.plain_symbol,
            expected.mathtext_symbol,
            expected.categories,
            expected.visibility,
        )


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


def test_every_controlled_fact_renders_without_its_python_field_name(
    catalog: ModuleType,
) -> None:
    """Deleting a catalog symbol must not expose a snake-case identifier in plain output."""
    for definition in catalog.FACT_DEFINITIONS:
        if definition.visibility_policy.value == "never":
            continue
        if definition.categorical_formatter is not None:
            fact = TextFact(
                definition.key,
                definition.categorical_formatter.values[0].raw,
                FieldImportance.ESSENTIAL,
            )
        elif definition.precision is not None:
            fact = DecimalFact(definition.key, 1.25, FieldImportance.ESSENTIAL)
        else:
            fact = IntegerFact(definition.key, 2, FieldImportance.ESSENTIAL)

        rendered = render_facts((fact,), NotationPolicy.default(RenderTarget.PLAIN))
        assert definition.key.value not in rendered
        assert "_" not in rendered


def test_mathtext_uses_catalog_markup_not_raw_identifiers(catalog: ModuleType) -> None:
    """MathText underscores come from controlled notation, never Python identifiers."""
    for definition in catalog.FACT_DEFINITIONS:
        if definition.visibility_policy.value == "never":
            continue
        if definition.categorical_formatter is not None:
            fact = TextFact(
                definition.key,
                definition.categorical_formatter.values[0].raw,
                FieldImportance.ESSENTIAL,
            )
        elif definition.precision is not None:
            fact = DecimalFact(definition.key, 1.25, FieldImportance.ESSENTIAL)
        else:
            fact = IntegerFact(definition.key, 2, FieldImportance.ESSENTIAL)

        rendered = render_facts((fact,), NotationPolicy.default(RenderTarget.MATHTEXT))
        if "_" in definition.key.value:
            assert definition.key.value not in rendered
        assert rendered.startswith(f"${definition.mathtext_symbol}=")


def test_unknown_fact_key_is_rejected(catalog: ModuleType) -> None:
    """Removing raw-key fallback must turn an unregistered key into an explicit error."""
    key = FieldKey("unregistered_scientific_parameter")
    fact = IntegerFact(key, 3, FieldImportance.ESSENTIAL)

    with pytest.raises(catalog.UnknownScientificFactError):
        catalog.fact_definition(key)
    with pytest.raises(catalog.UnknownScientificFactError):
        render_facts((fact,), NotationPolicy.default())


def test_unknown_categorical_value_is_rejected(catalog: ModuleType) -> None:
    """A category typo must not leak its raw enum spelling into presentation."""
    fact = TextFact(catalog.WEIGHTING, "unknown_weighting", FieldImportance.ESSENTIAL)

    with pytest.raises(catalog.UnknownCategoricalValueError):
        render_facts((fact,), NotationPolicy.default())


@pytest.fixture
def full_bamg_description(catalog: ModuleType) -> ScientificDescription:
    """All configured BAMG algorithm facts from the approved display contract."""
    essential = FieldImportance.ESSENTIAL
    return ScientificDescription(
        base_name="BAMG",
        facts=(
            IntegerFact(catalog.MULTIGRID_LEVELS, 10, essential),
            IntegerFact(catalog.RELAXATION_TEST_VECTORS, 8, essential),
            IntegerFact(catalog.RELAXATION_SWEEPS, 4, essential),
            IntegerFact(catalog.BOOTSTRAP_CYCLES, 2, essential),
            IntegerFact(catalog.COMPATIBLE_RELAXATION_SWEEPS, 5, essential),
            DecimalFact(catalog.COMPATIBLE_RELAXATION_TOLERANCE, 0.7, essential),
            DecimalFact(catalog.ALGEBRAIC_DISTANCE_THRESHOLD, 0.5, essential),
            IntegerFact(catalog.CALIBER, 4, essential),
            DecimalFact(catalog.CALIBER_GROWTH_PENALTY, 1.5, essential),
            TextFact(catalog.USE_LSR, "on", essential),
            IntegerFact(catalog.MAX_COARSE_DIMENSION, 10, essential),
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
