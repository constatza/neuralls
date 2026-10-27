"""Plots render typed method presentations: symbols and numbers, never ids, never dropped facts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from neuralls.composition.assignments.presentation_preflight import prepare_comparison_presentation
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    DatasetQualifiedMethodPresentation,
    MissingMethodPresentationError,
    PreparedComparisonPresentation,
    StandaloneMethodPresentation,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.render import (
    IndistinguishableMethodsError,
    render_distinct_methods,
)
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    IC0PreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting.figure_text import (
    EditorialTitle,
    Legend,
    LegendPlacement,
    method_legend,
    wrap_at_boundaries,
)
from neuralls.platform.reporting.plots import plot_convergence_comparison
from neuralls.shared.scientific_description.catalog import DROP_TOLERANCE, make_fact
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import (
    LEGEND_PROFILE,
    NotationPolicy,
    RenderProfile,
    RenderTarget,
)
from neuralls.shared.scientific_description.render import RenderedScientificDescription
from tests.dataset_support import build_test_dataset
from tests.presentation_support import pod_spec

MATHTEXT = NotationPolicy.default(RenderTarget.MATHTEXT)
PLAIN = NotationPolicy.default(RenderTarget.PLAIN)


@pytest.fixture
def second_pod_dataset_dir(
    tmp_path: Path, atomic_source: SourceSpec, atomic_spec: DatasetSpec
) -> Path:
    """A second fit dataset, generated with fewer samples so its qualifier differs."""
    target = tmp_path / "pod-fit-other"
    fewer = replace(
        atomic_spec, mixture=replace(atomic_spec.mixture, counts={"gaussian_forward": 4})
    )
    build_test_dataset(atomic_source, fewer, str(target))
    return target


@pytest.fixture
def acceptance_configs(
    pod_dataset_dir: Path, second_pod_dataset_dir: Path
) -> tuple[PreconditionerConfig, ...]:
    """Every method family of the acceptance figure, two variants where a fact separates them."""
    return (
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI),
        IC0PreconditionerConfig(),
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.25)),
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.5)),
        AdaptiveSAPreconditionerConfig(num_candidates=3),
        AdaptiveSAPreconditionerConfig(num_candidates=7),
        BootstrapAMGPreconditionerConfig(),
        *(
            pod_spec(directory, rank, assignment=f"fit-{index}-{rank}")
            for index, directory in enumerate((pod_dataset_dir, second_pod_dataset_dir))
            for rank in (10, 100)
        ),
    )


@pytest.fixture
def acceptance_cohort(
    acceptance_configs: tuple[PreconditionerConfig, ...],
    dataset_presentation: DatasetPresentation,
) -> PreparedComparisonPresentation:
    return prepare_comparison_presentation(
        acceptance_configs, ComparisonScopePresentation(dataset_presentation)
    )


def _result(name: str) -> CGComparisonResult:
    return CGComparisonResult(
        x=np.zeros(2),
        converged=True,
        iterations=3,
        residual=1e-9,
        residual_abs=1e-9,
        residual_history_rel=[1.0, 0.1, 1e-9],
        residual_history_abs=[1.0, 0.1, 1e-9],
        preconditioner=name,
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        setup_cost=1.0,
        solve_time_seconds=0.5,
    )


def _legend_of_cohort(cohort: PreparedComparisonPresentation, policy: NotationPolicy) -> Legend:
    return method_legend({m.method_id: m.compact for m in cohort.methods}, policy)


def _draw(cohort: PreparedComparisonPresentation, policy: NotationPolicy) -> Figure:
    results = {m.method_id: _result(m.method_id) for m in cohort.methods}
    with patch("neuralls.platform.reporting.plots.plt.close"):
        plot_convergence_comparison(
            results,
            legend=_legend_of_cohort(cohort, policy),
            title=EditorialTitle("Acceptance"),
        )
    return plt.gcf()


def _legend_texts(figure: Figure) -> list[str]:
    legend = figure.axes[0].get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


STANDALONE_LEGEND = [
    "Identity",
    "Jacobi",
    "IC0 (τ=0)",
    "AMG (L=2, θ=0.25, ν₁=2, ν₂=2, ωₛ=auto)",
    "AMG (L=2, θ=0.5, ν₁=2, ν₂=2, ωₛ=auto)",
    "aSA (L=2, k=3, nᵢₜ=5, cₘₐₓ=10, θ=0, ω=1.33)",
    "aSA (L=2, k=7, nᵢₜ=5, cₘₐₓ=10, θ=0, ω=1.33)",
    "BAMG (L=10, kᵣ=8, η=4, nᵦ=2, ν=5, δ=0.7, θₐd=0.5, q=4,\nγ=1.5, LSR=on, cₘₐₓ=10)",
]
POD_LEGEND = [
    "POD-2G (r=10, ν₁=2, ν₂=2, ωₛ=auto) |\nGaussian Solutions (μ=0, σ=1, N=8)",
    "POD-2G (r=100, ν₁=2, ν₂=2, ωₛ=auto) |\nGaussian Solutions (μ=0, σ=1, N=8)",
    "POD-2G (r=10, ν₁=2, ν₂=2, ωₛ=auto) |\nGaussian Solutions (μ=0, σ=1, N=4)",
    "POD-2G (r=100, ν₁=2, ν₂=2, ωₛ=auto) |\nGaussian Solutions (μ=0, σ=1, N=4)",
]
MATHTEXT_LEGEND = [
    "Identity",
    "Jacobi",
    "IC0 ($\\tau=0$)",
    "AMG ($L=2$, $\\theta=0.25$, $\\nu_1=2$, $\\nu_2=2$, $\\omega_s=\\mathrm{auto}$)",
    "AMG ($L=2$, $\\theta=0.5$, $\\nu_1=2$, $\\nu_2=2$, $\\omega_s=\\mathrm{auto}$)",
    "aSA ($L=2$, $k=3$, $n_{\\mathrm{it}}=5$, $c_{\\max}=10$, $\\theta=0$, $\\omega=1.33$)",
    "aSA ($L=2$, $k=7$, $n_{\\mathrm{it}}=5$, $c_{\\max}=10$, $\\theta=0$, $\\omega=1.33$)",
    "BAMG ($L=10$, $k_r=8$, $\\eta=4$, $n_b=2$, $\\nu=5$, $\\delta=0.7$, $\\theta_{\\mathrm{ad}}=0.5$, $q=4$,\n$\\gamma=1.5$, $\\mathrm{LSR}=\\mathrm{on}$, $c_{\\max}=10$)",
    "POD-2G ($r=10$, $\\nu_1=2$, $\\nu_2=2$, $\\omega_s=\\mathrm{auto}$) |\nGaussian Solutions ($\\mu=0$, $\\sigma=1$, $N=8$)",
    "POD-2G ($r=100$, $\\nu_1=2$, $\\nu_2=2$, $\\omega_s=\\mathrm{auto}$) |\nGaussian Solutions ($\\mu=0$, $\\sigma=1$, $N=8$)",
    "POD-2G ($r=10$, $\\nu_1=2$, $\\nu_2=2$, $\\omega_s=\\mathrm{auto}$) |\nGaussian Solutions ($\\mu=0$, $\\sigma=1$, $N=4$)",
    "POD-2G ($r=100$, $\\nu_1=2$, $\\nu_2=2$, $\\omega_s=\\mathrm{auto}$) |\nGaussian Solutions ($\\mu=0$, $\\sigma=1$, $N=4$)",
]


def test_acceptance_legend_shows_symbols_and_numbers_and_no_ids(
    acceptance_cohort: PreparedComparisonPresentation,
) -> None:
    figure = _draw(acceptance_cohort, MATHTEXT)
    texts = _legend_texts(figure)
    plt.close(figure)

    assert len(acceptance_cohort.methods) == 12
    assert texts == MATHTEXT_LEGEND


def test_each_pod_label_names_its_fit_dataset_exactly_once(
    acceptance_cohort: PreparedComparisonPresentation,
) -> None:
    figure = _draw(acceptance_cohort, PLAIN)
    texts = _legend_texts(figure)
    plt.close(figure)

    assert texts[len(STANDALONE_LEGEND) :] == POD_LEGEND


def test_adaptive_sa_variants_differ_by_their_facts_without_a_key_suffix(
    acceptance_cohort: PreparedComparisonPresentation,
) -> None:
    figure = _draw(acceptance_cohort, PLAIN)
    texts = _legend_texts(figure)
    plt.close(figure)

    assert texts == [*STANDALONE_LEGEND, *POD_LEGEND]


def test_a_method_without_a_presentation_raises_instead_of_using_its_key(
    acceptance_cohort: PreparedComparisonPresentation,
) -> None:
    legend = method_legend({m.method_id: m.compact for m in acceptance_cohort.methods[:2]}, PLAIN)
    stray = acceptance_cohort.methods[2].method_id

    with pytest.raises(MissingMethodPresentationError) as raised:
        legend.label(stray)
    with pytest.raises(MissingMethodPresentationError):
        plot_convergence_comparison({stray: _result(stray)}, legend=legend)

    assert raised.value.method_id == stray
    plt.close("all")


def _ilut(tolerance: float) -> ConfiguredMethodPresentation:
    return StandaloneMethodPresentation(
        ScientificDescription("ILUT", facts=(make_fact(DROP_TOLERANCE, tolerance),))
    )


def test_values_that_round_alike_are_rendered_with_more_digits_until_distinct() -> None:
    methods = {"a": _ilut(0.0012), "b": _ilut(0.0014)}
    at_catalog_precision = {
        name: method_legend({name: method}, PLAIN).label(name) for name, method in methods.items()
    }
    assert at_catalog_precision["a"] == at_catalog_precision["b"]  # one digit: both 0.001

    legend = method_legend(methods, PLAIN)

    assert legend.label("a") == "ILUT (τ=0.0012)"
    assert legend.label("b") == "ILUT (τ=0.0014)"


def test_values_that_stay_identical_at_full_precision_are_rejected() -> None:
    with pytest.raises(IndistinguishableMethodsError):
        render_distinct_methods({"a": _ilut(0.5), "b": _ilut(0.5)}, LEGEND_PROFILE, PLAIN)


def test_the_length_limit_wraps_lines_and_never_drops_a_fact(
    acceptance_cohort: PreparedComparisonPresentation,
) -> None:
    pod = next(
        m
        for m in acceptance_cohort.methods
        if isinstance(m.configured, DatasetQualifiedMethodPresentation)
    )
    unbounded = RenderProfile(name="unbounded", include_context=False)
    full = render_distinct_methods({"pod": pod.compact}, unbounded, PLAIN).texts["pod"].text

    wrapped = method_legend({"pod": pod.compact}, PLAIN).label("pod")

    assert len(wrapped.split("\n")) > 1
    assert wrapped.replace(",\n", ", ").replace("\n", " ") == full
    limit = LEGEND_PROFILE.max_length
    assert limit is not None
    assert all(len(line) <= limit for line in wrapped.split("\n"))


def test_wrapping_breaks_only_at_the_given_boundaries() -> None:
    drawn = RenderedScientificDescription("$a=1$, $b=2$, $c=3$", (7, 14))
    visible = RenderedScientificDescription("a=1, b=2, c=3", (5, 10))

    assert wrap_at_boundaries(drawn, visible, limit=7) == "$a=1$,\n$b=2$,\n$c=3$"
    assert wrap_at_boundaries(drawn, visible, limit=100) == drawn.text
    with pytest.raises(ValueError):
        wrap_at_boundaries(drawn, RenderedScientificDescription("a=1, b=2", (5,)), limit=3)


def test_a_long_legend_moves_outside_the_axes_and_keeps_every_fact(
    acceptance_cohort: PreparedComparisonPresentation,
) -> None:
    legend = _legend_of_cohort(acceptance_cohort, MATHTEXT)
    unbounded = RenderProfile(name="unbounded", include_context=False)
    facts_of = render_distinct_methods(
        {m.method_id: m.compact for m in acceptance_cohort.methods}, unbounded, MATHTEXT
    ).texts

    assert legend.layout.placement is LegendPlacement.OUTSIDE_RIGHT
    assert legend.layout.columns >= 1
    for method in acceptance_cohort.methods:
        flat = legend.label(method.method_id).replace(",\n", ", ").replace("\n", " ")
        assert flat == facts_of[method.method_id].text

    figure = _draw(acceptance_cohort, MATHTEXT)
    drawn = _legend_texts(figure)
    plt.close(figure)
    assert drawn == [legend.label(m.method_id) for m in acceptance_cohort.methods]


def test_a_short_legend_stays_inside(legend_of: Callable[..., Legend]) -> None:
    layout = legend_of("a", "b").layout

    assert layout == replace(layout, placement=LegendPlacement.INSIDE, columns=1)


@pytest.mark.real_render
def test_the_acceptance_figure_encodes_a_real_png(
    acceptance_cohort: PreparedComparisonPresentation, tmp_path: Path
) -> None:
    results = {m.method_id: _result(m.method_id) for m in acceptance_cohort.methods}
    target = tmp_path / "acceptance.png"

    plot_convergence_comparison(
        results,
        legend=_legend_of_cohort(acceptance_cohort, MATHTEXT),
        title=EditorialTitle("Acceptance"),
        save_path=target,
    )

    assert target.read_bytes().startswith(b"\x89PNG")
    assert target.stat().st_size > 10_000


def test_a_standalone_method_label_has_no_dataset_separator(
    acceptance_cohort: PreparedComparisonPresentation,
) -> None:
    figure = _draw(acceptance_cohort, PLAIN)
    texts = _legend_texts(figure)
    plt.close(figure)

    assert texts[: len(STANDALONE_LEGEND)] == STANDALONE_LEGEND
