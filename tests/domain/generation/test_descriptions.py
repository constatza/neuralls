"""Tests for generation strategy descriptions (``docs/plan.md`` Phase 5)."""

from __future__ import annotations

import pytest

from neuralls.domain.generation.descriptions import (
    RESIDUAL_TRACE_STRATEGY_NAMES,
    describe_generation_result,
    describe_residual_trace,
)
from neuralls.shared.scientific_description import LOG_PROFILE, NotationPolicy, render


def test_residuals_strategy_renders_canonical_example() -> None:
    """Matches the plan's own worked example exactly."""
    description = describe_residual_trace("residuals", iteration_count=15, sample_count=500)
    rendered = render(description, LOG_PROFILE, NotationPolicy.default())
    assert rendered.text == "CG Residuals (k=15, N=500)"


def test_gaussian_residuals_strategy_uses_distinct_base_name() -> None:
    """The Gaussian variant renders a distinct, non-colliding base name."""
    description = describe_residual_trace(
        "gaussian_residuals", iteration_count=10, sample_count=250
    )
    rendered = render(description, LOG_PROFILE, NotationPolicy.default())
    assert rendered.text == "Gaussian CG Residuals (k=10, N=250)"


def test_unknown_residual_trace_name_is_rejected() -> None:
    """An unrecognized strategy name never becomes a base name."""
    with pytest.raises(ValueError, match="Unsupported generation strategy"):
        describe_residual_trace("future_strategy", iteration_count=1, sample_count=1)


@pytest.mark.parametrize(
    ("strategy_name", "expected"),
    [
        ("krylov", "Krylov Subspace Solutions (N=1000)"),
        ("rhs_archive", "Archived RHS Solves (N=1000)"),
        ("solution_archive", "Archived Solutions (N=1000)"),
        ("validated_archive", "Validated Archive Pairs (N=1000)"),
        ("scaled_solutions", "Scaled Archived Solutions (N=1000)"),
        ("sparse_rhs", "Sparse RHS Solves (N=1000)"),
        ("eigenvector_forward", "Eigenvector Solutions (N=1000)"),
        ("eigenvector_inverse", "Eigenvector RHS Solves (N=1000)"),
        ("gaussian_forward", "Gaussian Solutions (N=1000)"),
        ("gaussian_inverse", "Gaussian RHS Solves (N=1000)"),
        ("uniform_forward", "Uniform Solutions (N=1000)"),
        ("uniform_inverse", "Uniform RHS Solves (N=1000)"),
        ("constant_forward", "Constant Solutions (N=1000)"),
        ("constant_inverse", "Constant RHS Solves (N=1000)"),
        ("neutral_ones", "Ones Solutions (N=1000)"),
        ("smoother_filtered_probes", "Jacobi-Filtered Probes (N=1000)"),
        ("sinusoidal_field", "Sinusoidal RHS Solves (N=1000)"),
        ("random", "Random Normal Solutions (N=1000)"),
        ("normal", "Random Normal Solutions (N=1000)"),
    ],
)
def test_describe_generation_result_covers_every_registered_strategy(
    strategy_name: str, expected: str
) -> None:
    """Every non-residual-trace strategy renders a name that says what it generates."""
    description = describe_generation_result(strategy_name, sample_count=1000)
    assert render(description, LOG_PROFILE, NotationPolicy.default()).text == expected


def test_describe_generation_result_rejects_unknown_strategy() -> None:
    """A strategy without a controlled name fails instead of echoing its identifier."""
    with pytest.raises(ValueError, match="Unsupported generation strategy"):
        describe_generation_result("future_strategy", sample_count=5)


def test_residual_trace_strategy_names_are_excluded_from_generic_dispatch() -> None:
    """The residual-trace pair is exactly what `runner.py` must skip, no more, no less."""
    assert RESIDUAL_TRACE_STRATEGY_NAMES == {"residuals", "gaussian_residuals"}
