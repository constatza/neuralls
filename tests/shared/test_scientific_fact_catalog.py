"""Contract tests for the controlled scientific-fact catalog."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
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
        ExpectedFactDefinition(
            "snapshot_count", "algorithmic", "configured", 310, None, "Nₛ", "N_s"
        ),
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
            "model_architecture", "algorithmic", "configured", 380, None, "arch", "\\mathrm{arch}"
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
        ExpectedFactDefinition(
            "start_iteration",
            "scheduling",
            "configured",
            410,
            None,
            "k₀",
            "k_0",
            visibility="when_non_default",
        ),
        ExpectedFactDefinition(
            "iteration_limit",
            "scheduling",
            "configured",
            420,
            None,
            "kₘₐₓ",
            "k_{\\max}",
            visibility="when_non_default",
        ),
        ExpectedFactDefinition(
            "fallback",
            "scheduling",
            "configured",
            430,
            None,
            "fb",
            "\\mathrm{fb}",
            categories=(
                ("identity", "Identity", "\\mathrm{Identity}"),
                ("jacobi", "Jacobi", "\\mathrm{Jacobi}"),
                ("ilu", "ILU", "\\mathrm{ILU}"),
                ("ic0", "IC(0)", "\\mathrm{IC(0)}"),
                ("neural", "Neural", "\\mathrm{Neural}"),
                ("amg", "AMG", "\\mathrm{AMG}"),
                ("neural_amg", "Neural AMG", "\\mathrm{Neural\\ AMG}"),
                ("adaptive_sa_amg", "aSA", "\\mathrm{aSA}"),
                ("bootstrap_amg", "BAMG", "\\mathrm{BAMG}"),
            ),
            visibility="when_non_default",
        ),
        ExpectedFactDefinition(
            "realized_levels", "algorithmic", "realized", 440, None, "Lᵣ", "L_r"
        ),
        ExpectedFactDefinition(
            "window_start",
            "algorithmic",
            "configured",
            450,
            None,
            "kₛ",
            "k_s",
        ),
        ExpectedFactDefinition(
            "window_stride",
            "algorithmic",
            "configured",
            460,
            None,
            "Δk",
            "\\Delta k",
        ),
        ExpectedFactDefinition(
            "eigenvalue_selection",
            "algorithmic",
            "configured",
            470,
            None,
            "λ",
            "\\lambda",
            categories=(
                ("smallest", "min", "\\mathrm{min}"),
                ("largest", "max", "\\mathrm{max}"),
                ("random", "rand", "\\mathrm{rand}"),
            ),
        ),
        ExpectedFactDefinition(
            "eigenvector_count",
            "algorithmic",
            "configured",
            480,
            None,
            "nₑ",
            "n_e",
        ),
        ExpectedFactDefinition(
            "eigenvector_inclusion",
            "algorithmic",
            "configured",
            490,
            None,
            "EV",
            "\\mathrm{EV}",
            categories=(
                ("off", "off", "\\mathrm{off}"),
                ("on", "on", "\\mathrm{on}"),
            ),
        ),
        ExpectedFactDefinition(
            "solver_kind",
            "algorithmic",
            "configured",
            500,
            None,
            "S",
            "S",
            categories=(
                ("direct", "direct", "\\mathrm{direct}"),
                ("cg", "CG", "\\mathrm{CG}"),
            ),
        ),
        ExpectedFactDefinition(
            "solver_relative_tolerance",
            "algorithmic",
            "configured",
            510,
            3,
            "εᵣ",
            "\\epsilon_r",
        ),
        ExpectedFactDefinition(
            "solver_absolute_tolerance",
            "algorithmic",
            "configured",
            520,
            3,
            "εₐ",
            "\\epsilon_a",
        ),
        ExpectedFactDefinition(
            "solver_max_iterations",
            "algorithmic",
            "configured",
            530,
            None,
            "Kₘₐₓ",
            "K_{\\max}",
        ),
        ExpectedFactDefinition(
            "solver_positive_definite",
            "algorithmic",
            "configured",
            540,
            None,
            "SPD",
            "\\mathrm{SPD}",
            categories=(
                ("off", "off", "\\mathrm{off}"),
                ("on", "on", "\\mathrm{on}"),
            ),
        ),
        ExpectedFactDefinition(
            "constant_value",
            "algorithmic",
            "configured",
            550,
            3,
            "v",
            "v",
        ),
        ExpectedFactDefinition(
            "frequency",
            "algorithmic",
            "configured",
            560,
            3,
            "ω",
            "\\omega",
        ),
        ExpectedFactDefinition(
            "rhs_scale",
            "algorithmic",
            "configured",
            570,
            3,
            "‖b‖",
            "\\Vert b\\Vert",
        ),
        ExpectedFactDefinition(
            "solution_scale",
            "algorithmic",
            "configured",
            580,
            3,
            "α",
            "\\alpha",
        ),
        ExpectedFactDefinition(
            "verification_tolerance",
            "algorithmic",
            "configured",
            590,
            3,
            "εᵥ",
            "\\epsilon_v",
        ),
        ExpectedFactDefinition(
            "sparse_support",
            "algorithmic",
            "configured",
            600,
            None,
            "nnz",
            "\\mathrm{nnz}",
        ),
        ExpectedFactDefinition(
            "mask_density",
            "algorithmic",
            "configured",
            610,
            3,
            "ρ",
            "\\rho",
        ),
    )


def test_catalog_is_exhaustive_and_pins_every_definition(
    catalog: ModuleType, expected_definitions: tuple[ExpectedFactDefinition, ...]
) -> None:
    """A wrong/missing role, phase, order, symbol, precision, category, or visibility must fail."""
    actual = {definition.key.value: definition for definition in catalog.FACT_DEFINITIONS}
    assert len(catalog.FACT_DEFINITIONS) == len(expected_definitions)
    assert set(actual) == {expected.key for expected in expected_definitions}

    for expected in expected_definitions:
        definition = actual[expected.key]
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


def test_compact_by_policy_is_pinned_for_every_visibility() -> None:
    assert {v for v in FactVisibility if is_compact_by_policy(v)} == {
        FactVisibility.ALWAYS,
        FactVisibility.WHEN_NON_DEFAULT,
    }
