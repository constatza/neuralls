"""Tests for the pure scientific-description core."""

from __future__ import annotations

import pytest

from neuralls.shared.scientific_description.catalog import make_fact
from neuralls.shared.scientific_description.models import (
    DecimalFact,
    FieldKey,
    ScientificDescription,
)
from neuralls.shared.scientific_description.policy import (
    LEGEND_PROFILE,
    LOG_PROFILE,
    NotationPolicy,
    RenderProfile,
    RenderTarget,
    SymbolSpec,
)
from neuralls.shared.scientific_description.render import render, render_facts

ITERATION_COUNT = FieldKey("iteration_count")
SAMPLE_COUNT = FieldKey("sample_count")
WEIGHTING = FieldKey("weighting")


@pytest.fixture
def cg_residuals_description() -> ScientificDescription:
    """``CG Residuals`` with one always-visible algorithmic context fact."""
    return ScientificDescription(
        base_name="CG Residuals",
        facts=(
            make_fact(ITERATION_COUNT, 15),
            make_fact(SAMPLE_COUNT, 500),
        ),
        context_facts=(make_fact(WEIGHTING, "uniform"),),
    )


def test_field_key_rejects_non_snake_case() -> None:
    """A FieldKey must be a validated snake_case identifier."""
    with pytest.raises(ValueError):
        FieldKey("Not Snake Case")


def test_renders_default_plain_label(cg_residuals_description: ScientificDescription) -> None:
    """The canonical example renders exactly as specified in the plan."""
    rendered = render(cg_residuals_description, LOG_PROFILE, NotationPolicy.default())
    assert rendered.text == "CG Residuals (k=15, N=500, w=uniform)"


def test_legend_profile_retains_always_visible_algorithmic_context(
    cg_residuals_description: ScientificDescription,
) -> None:
    """Compact profiles cannot hide algorithmic facts marked always-visible."""
    rendered = render(cg_residuals_description, LEGEND_PROFILE, NotationPolicy.default())
    assert rendered.text == "CG Residuals (k=15, N=500, w=uniform)"


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (RenderTarget.PLAIN, "CG Residuals (k=15, N=500, w=uniform)"),
        (
            RenderTarget.MATHTEXT,
            "CG Residuals ($k=15$, $N=500$, $w=\\mathrm{uniform}$)",
        ),
        (
            RenderTarget.LATEX,
            "CG Residuals ($k=15$, $N=500$, $w=\\mathrm{uniform}$)",
        ),
    ],
)
def test_supports_all_render_targets(
    cg_residuals_description: ScientificDescription, target: RenderTarget, expected: str
) -> None:
    """Plain, MathText, and LaTeX targets all render from the same facts."""
    rendered = render(cg_residuals_description, LEGEND_PROFILE, NotationPolicy.default(target))
    assert rendered.text == expected


def test_symbol_override(cg_residuals_description: ScientificDescription) -> None:
    """A caller-supplied symbol table overrides the default plain symbol."""
    policy = NotationPolicy.default().with_overrides(
        {ITERATION_COUNT: SymbolSpec(plain="k_CG", math="k_{\\mathrm{CG}}")}
    )
    rendered = render(cg_residuals_description, LEGEND_PROFILE, policy)
    assert rendered.text == "CG Residuals (k_CG=15, N=500, w=uniform)"


def test_precision_override() -> None:
    """A per-key precision override controls decimal formatting."""

    standard_deviation = FieldKey("standard_deviation")
    description = ScientificDescription(
        base_name="Gaussian LHS",
        facts=(make_fact(standard_deviation, 0.0123456),),
    )
    profile = RenderProfile(
        name="detailed", include_context=True, precision={standard_deviation: 2}
    )
    rendered = render(description, profile, NotationPolicy.default())
    assert rendered.text == "Gaussian LHS (σ=0.012)"


def test_length_budget_returns_breakpoints_without_dropping_facts(
    cg_residuals_description: ScientificDescription,
) -> None:
    """A narrow profile changes layout hints, never the selected scientific facts."""
    profile = RenderProfile(name="narrow", include_context=True, max_length=10)

    rendered = render(cg_residuals_description, profile, NotationPolicy.default())

    assert rendered.text == "CG Residuals (k=15, N=500, w=uniform)"
    assert len(rendered.preferred_wrap_boundaries) == 2


def test_render_facts_has_no_base_name_or_parens() -> None:
    """``render_facts`` joins facts flatly, for callers composing their own base name."""
    facts = (
        make_fact(ITERATION_COUNT, 15),
        make_fact(SAMPLE_COUNT, 500),
    )
    assert render_facts(facts, NotationPolicy.default()) == "k=15, N=500"


def test_render_facts_applies_precision_override() -> None:
    """``render_facts`` accepts the same per-key precision override as ``render``."""

    theta = FieldKey("threshold")
    facts = (make_fact(theta, 0.25),)
    rendered = render_facts(facts, NotationPolicy.default(), precision={theta: 2})
    assert rendered == "θ=0.25"


def test_description_rejects_empty_base_name() -> None:
    """A ``ScientificDescription`` must have a non-empty base name."""
    with pytest.raises(ValueError):
        ScientificDescription(base_name="")


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_decimal_fact_rejects_non_finite_values(value: float) -> None:
    """A non-finite numerical fact cannot enter the semantic model."""
    with pytest.raises(ValueError, match="finite"):
        DecimalFact(ITERATION_COUNT, value)


def test_length_budget_is_measured_on_visible_width_not_markup() -> None:
    """`$\\theta=0.25$` is markup: the same facts must survive the same budget in every target."""
    key_a, key_b = FieldKey("threshold"), FieldKey("damping")
    description = ScientificDescription(
        base_name="AMG",
        facts=(
            make_fact(key_a, 0.25),
            make_fact(key_b, 0.67),
        ),
    )
    plain = NotationPolicy.default(RenderTarget.PLAIN)
    mathtext = NotationPolicy.default(RenderTarget.MATHTEXT)
    visible_width = len(render(description, RenderProfile("p", False), plain).text)
    profile = RenderProfile(name="tight", include_context=False, max_length=visible_width)

    plain_rendered = render(description, profile, plain)
    mathtext_rendered = render(description, profile, mathtext)
    assert plain_rendered.text == "AMG (θ=0.25, ω=0.67)"
    assert mathtext_rendered.text == r"AMG ($\theta=0.25$, $\omega=0.67$)"
    assert plain_rendered.preferred_wrap_boundaries == ()
    assert mathtext_rendered.preferred_wrap_boundaries == ()
