"""Tests for the pure scientific-description core (``docs/plan.md`` Phase 1)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from neuralls.shared.scientific_description import (
    DETAILED_PROFILE,
    LEGEND_PROFILE,
    LOG_PROFILE,
    DecimalFact,
    FieldImportance,
    FieldKey,
    IntegerFact,
    NotationPolicy,
    RenderProfile,
    RenderTarget,
    ScientificDescription,
    SymbolSpec,
    TextFact,
    render,
    render_facts,
)

ITERATION_COUNT = FieldKey("iteration_count")
SAMPLE_COUNT = FieldKey("sample_count")
WEIGHTING = FieldKey("weighting")


@pytest.fixture
def cg_residuals_description() -> ScientificDescription:
    """``CG Residuals (k=15, N=500)`` with one dropped-first context fact."""
    return ScientificDescription(
        base_name="CG Residuals",
        facts=(
            IntegerFact(key=ITERATION_COUNT, value=15, importance=FieldImportance.ESSENTIAL),
            IntegerFact(key=SAMPLE_COUNT, value=500, importance=FieldImportance.ESSENTIAL),
        ),
        context_facts=(
            TextFact(key=WEIGHTING, value="uniform", importance=FieldImportance.VARIANT),
        ),
    )


def test_field_key_rejects_non_snake_case() -> None:
    """A FieldKey must be a validated snake_case identifier."""
    with pytest.raises(ValueError, match="snake_case"):
        FieldKey("Not Snake Case")


def test_renders_default_plain_label(cg_residuals_description: ScientificDescription) -> None:
    """The canonical example renders exactly as specified in the plan."""
    rendered = render(cg_residuals_description, LOG_PROFILE, NotationPolicy.default())
    assert rendered == "CG Residuals (k=15, N=500, weighting=uniform)"


def test_legend_profile_excludes_context(cg_residuals_description: ScientificDescription) -> None:
    """``include_context=False`` drops context facts entirely."""
    rendered = render(cg_residuals_description, LEGEND_PROFILE, NotationPolicy.default())
    assert rendered == "CG Residuals (k=15, N=500)"


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (RenderTarget.PLAIN, "CG Residuals (k=15, N=500)"),
        (RenderTarget.MATHTEXT, "CG Residuals ($k=15$, $N=500$)"),
        (RenderTarget.LATEX, "CG Residuals ($k=15$, $N=500$)"),
    ],
)
def test_supports_all_render_targets(
    cg_residuals_description: ScientificDescription, target: RenderTarget, expected: str
) -> None:
    """Plain, MathText, and LaTeX targets all render from the same facts."""
    rendered = render(cg_residuals_description, LEGEND_PROFILE, NotationPolicy.default(target))
    assert rendered == expected


def test_symbol_override(cg_residuals_description: ScientificDescription) -> None:
    """A caller-supplied symbol table overrides the default plain symbol."""
    policy = NotationPolicy.default().with_overrides(
        {ITERATION_COUNT: SymbolSpec(plain="k_CG", math="k_{\\mathrm{CG}}")}
    )
    rendered = render(cg_residuals_description, LEGEND_PROFILE, policy)
    assert rendered == "CG Residuals (k_CG=15, N=500)"


def test_field_order_override(cg_residuals_description: ScientificDescription) -> None:
    """An explicit field order takes priority over the description's own order."""
    profile = RenderProfile(
        name="legend", include_context=False, field_order=(SAMPLE_COUNT, ITERATION_COUNT)
    )
    rendered = render(cg_residuals_description, profile, NotationPolicy.default())
    assert rendered == "CG Residuals (N=500, k=15)"


def test_precision_override() -> None:
    """A per-key precision override controls decimal formatting."""
    from neuralls.shared.scientific_description import DecimalFact

    standard_deviation = FieldKey("standard_deviation")
    description = ScientificDescription(
        base_name="Gaussian LHS",
        facts=(
            DecimalFact(
                key=standard_deviation, value=0.0123456, importance=FieldImportance.ESSENTIAL
            ),
        ),
    )
    profile = RenderProfile(
        name="detailed", include_context=True, precision={standard_deviation: 2}
    )
    rendered = render(description, profile, NotationPolicy.default())
    assert rendered == "Gaussian LHS (σ=0.012)"


def test_length_budget_drops_context_before_variant_before_essential() -> None:
    """Long labels shrink by priority, never by arbitrary truncation."""
    a, b, c = FieldKey("field_a"), FieldKey("field_b"), FieldKey("field_c")
    description = ScientificDescription(
        base_name="Mixed",
        facts=(
            TextFact(key=a, value="essential-value", importance=FieldImportance.ESSENTIAL),
            TextFact(key=b, value="variant-value", importance=FieldImportance.VARIANT),
        ),
        context_facts=(TextFact(key=c, value="context-value", importance=FieldImportance.VARIANT),),
    )
    full = render(description, DETAILED_PROFILE, NotationPolicy.default())
    assert full == "Mixed (field_a=essential-value, field_b=variant-value, field_c=context-value)"

    budget = RenderProfile(name="shrunk", include_context=True, max_length=len(full) - 1)
    shrunk = render(description, budget, NotationPolicy.default())
    assert "field_c" not in shrunk
    assert "field_b" in shrunk
    assert "field_a" in shrunk

    tighter = RenderProfile(name="tighter", include_context=True, max_length=10)
    tightest = render(description, tighter, NotationPolicy.default())
    assert "field_a" in tightest
    assert "field_b" not in tightest
    assert "field_c" not in tightest


def test_render_facts_has_no_base_name_or_parens() -> None:
    """``render_facts`` joins facts flatly, for callers composing their own base name."""
    facts = (
        IntegerFact(key=ITERATION_COUNT, value=15, importance=FieldImportance.ESSENTIAL),
        IntegerFact(key=SAMPLE_COUNT, value=500, importance=FieldImportance.ESSENTIAL),
    )
    assert render_facts(facts, NotationPolicy.default()) == "k=15, N=500"


def test_render_facts_applies_precision_override() -> None:
    """``render_facts`` accepts the same per-key precision override as ``render``."""
    from neuralls.shared.scientific_description import DecimalFact

    theta = FieldKey("threshold")
    facts = (DecimalFact(key=theta, value=0.25, importance=FieldImportance.ESSENTIAL),)
    rendered = render_facts(facts, NotationPolicy.default(), precision={theta: 2})
    assert rendered == "θ=0.25"


def test_description_rejects_empty_base_name() -> None:
    """A ``ScientificDescription`` must have a non-empty base name."""
    with pytest.raises(ValueError, match="base_name"):
        ScientificDescription(base_name="")


def test_package_imports_standard_library_only() -> None:
    """No module under ``scientific_description`` imports outside the stdlib."""
    package_dir = (
        Path(sys.modules["neuralls"].__file__).parent / "shared" / "scientific_description"
    )
    stdlib_names = set(sys.stdlib_module_names) | {"neuralls"}
    for source_file in package_dir.glob("*.py"):
        tree = ast.parse(source_file.read_text(), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                assert name in stdlib_names, f"{source_file.name} imports non-stdlib {name!r}"


def test_length_budget_is_measured_on_visible_width_not_markup() -> None:
    """`$\\theta=0.25$` is markup: the same facts must survive the same budget in every target."""
    key_a, key_b = FieldKey("threshold"), FieldKey("damping")
    description = ScientificDescription(
        base_name="AMG",
        facts=(
            DecimalFact(key_a, 0.25, FieldImportance.ESSENTIAL),
            DecimalFact(key_b, 0.67, FieldImportance.VARIANT),
        ),
    )
    plain = NotationPolicy.default(RenderTarget.PLAIN)
    mathtext = NotationPolicy.default(RenderTarget.MATHTEXT)
    visible_width = len(render(description, RenderProfile("p", False), plain))
    profile = RenderProfile(name="tight", include_context=False, max_length=visible_width)

    assert "ω" in render(description, profile, plain)
    assert "\\omega" in render(description, profile, mathtext)
