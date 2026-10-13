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
    assert (
        render(description, LOG_PROFILE, NotationPolicy.default()) == "CG Residuals (k=15, N=500)"
    )


def test_gaussian_residuals_strategy_uses_distinct_base_name() -> None:
    """The Gaussian variant renders a distinct, non-colliding base name."""
    description = describe_residual_trace(
        "gaussian_residuals", iteration_count=10, sample_count=250
    )
    rendered = render(description, LOG_PROFILE, NotationPolicy.default())
    assert rendered == "Gaussian CG Residuals (k=10, N=250)"


def test_unknown_strategy_name_falls_back_to_raw_name() -> None:
    """An unrecognized strategy name is used verbatim rather than raising."""
    description = describe_residual_trace("future_strategy", iteration_count=1, sample_count=1)
    assert description.base_name == "future_strategy"


@pytest.mark.parametrize(
    ("strategy_name", "expected"),
    [
        ("krylov", "Krylov Combinations (N=1000)"),
        ("rhs_archive", "RHS Archive (N=1000)"),
        ("solution_archive", "Solution Archive (N=1000)"),
        ("validated_archive", "Validated Archive (N=1000)"),
        ("scaled_solutions", "Scaled Solutions (N=1000)"),
        ("sparse_rhs", "Sparse RHS (N=1000)"),
        ("eigenvector_forward", "Eigenvector Forward (N=1000)"),
        ("eigenvector_inverse", "Eigenvector Inverse (N=1000)"),
        ("gaussian_forward", "Gaussian Forward (N=1000)"),
        ("gaussian_inverse", "Gaussian Inverse (N=1000)"),
        ("uniform_forward", "Uniform Forward (N=1000)"),
        ("uniform_inverse", "Uniform Inverse (N=1000)"),
        ("constant_forward", "Constant Forward (N=1000)"),
        ("constant_inverse", "Constant Inverse (N=1000)"),
        ("neutral_ones", "Neutral Ones (N=1000)"),
        ("smoother_filtered_probes", "Smoother-Filtered Probes (N=1000)"),
        ("sinusoidal_field", "Sinusoidal Field (N=1000)"),
        ("random", "Gaussian LHS (N=1000)"),
        ("normal", "Gaussian LHS (N=1000)"),
    ],
)
def test_describe_generation_result_covers_every_registered_strategy(
    strategy_name: str, expected: str
) -> None:
    """Every non-residual-trace strategy renders its own base name plus the realized N.

    Matches the user's own worked example shape, e.g. ``"Error Traces (N=1000)"``.
    """
    description = describe_generation_result(strategy_name, sample_count=1000)
    assert render(description, LOG_PROFILE, NotationPolicy.default()) == expected


def test_describe_generation_result_falls_back_for_unknown_strategy() -> None:
    """A future strategy added without updating the base-name table still renders, not raises."""
    description = describe_generation_result("future_strategy", sample_count=5)
    assert render(description, LOG_PROFILE, NotationPolicy.default()) == "future_strategy (N=5)"


def test_residual_trace_strategy_names_are_excluded_from_generic_dispatch() -> None:
    """The residual-trace pair is exactly what `runner.py` must skip, no more, no less."""
    assert RESIDUAL_TRACE_STRATEGY_NAMES == {"residuals", "gaussian_residuals"}
