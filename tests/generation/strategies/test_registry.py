"""Regression tests for package-level strategy registration."""

from __future__ import annotations

import numpy as np

import neuralls.domain.generation
from neuralls.domain.generation import run_generation
from neuralls.domain.generation.runner import _registry, strategy_supports_matrix_replacement

EXPECTED_STRATEGIES = {
    "random",
    "normal",
    "krylov",
    "residuals",
    "gaussian_residuals",
    "search_directions",
    "eigenvector_forward",
    "eigenvector_inverse",
    "rhs_archive",
    "solution_archive",
    "neutral_ones",
    "gaussian_forward",
    "gaussian_inverse",
    "uniform_forward",
    "uniform_inverse",
    "constant_forward",
    "constant_inverse",
    "validated_archive",
    "scaled_solutions",
    "sparse_rhs",
    "smoother_filtered_probes",
}


def test_all_strategy_modules_register_on_package_import() -> None:
    """All strategy names are available after importing neuralls.domain.generation."""
    assert neuralls.domain.generation.strategies is not None
    assert set(_registry._strategies) == EXPECTED_STRATEGIES


def test_gaussian_inverse_runs_after_package_import(
    spd_matrix: np.ndarray,
) -> None:
    """Gaussian inverse is reachable through the public dispatcher."""
    sample_count = 3
    result = run_generation(
        "gaussian_inverse",
        spd_matrix,
        cfg={"samples": sample_count, "seed": 0, "mu": 0.0, "sigma": 1.0},
    )

    assert result.rhs is not None
    assert result.solutions is not None
    assert result.rhs.shape == (sample_count, spd_matrix.shape[0])
    assert result.solutions.shape == (sample_count, spd_matrix.shape[0])


def test_removed_legacy_trace_names_fail(spd_matrix: np.ndarray) -> None:
    """Legacy trace strategy names are intentionally unsupported."""
    for legacy_name in (
        "cg_residual",
        "residual",
        "cg_residual_error",
        "residual_error",
    ):
        try:
            run_generation(legacy_name, spd_matrix, cfg={"samples": 1, "cg_iters": 1})
            assert False, f"{legacy_name} should be removed"
        except KeyError:
            pass


def test_matrix_replacement_capabilities_are_centrally_registered() -> None:
    """Replacement support is opt-in and queried through registry metadata."""
    assert strategy_supports_matrix_replacement("random") is True
    assert strategy_supports_matrix_replacement("gaussian_forward") is True
    assert strategy_supports_matrix_replacement("gaussian_residuals") is True
    assert strategy_supports_matrix_replacement("solution_archive") is False
    assert strategy_supports_matrix_replacement("neutral_ones") is False
