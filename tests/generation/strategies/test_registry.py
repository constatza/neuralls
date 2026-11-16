"""Regression tests for package-level strategy registration."""

from __future__ import annotations

import numpy as np

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.runner import (
    ArchiveStrategyRegistration,
    MatrixStrategyRegistration,
    SingleRhsStrategyRegistration,
    StrategyRegistration,
    _registry,
)
from tests.fakes.archive_source import InMemoryArchiveSource
from tests.generation_support import solved_rows

ARCHIVE_SOURCE = InMemoryArchiveSource()

EXPECTED_STRATEGIES: dict[str, type[StrategyRegistration]] = {
    "constant_forward": MatrixStrategyRegistration,
    "constant_inverse": MatrixStrategyRegistration,
    "eigenvector_forward": MatrixStrategyRegistration,
    "eigenvector_inverse": MatrixStrategyRegistration,
    "gaussian_forward": MatrixStrategyRegistration,
    "gaussian_inverse": MatrixStrategyRegistration,
    "cg_errors": SingleRhsStrategyRegistration,
    "krylov": MatrixStrategyRegistration,
    "neutral_ones": MatrixStrategyRegistration,
    "normal": MatrixStrategyRegistration,
    "random": MatrixStrategyRegistration,
    "residuals": SingleRhsStrategyRegistration,
    "rhs_archive": ArchiveStrategyRegistration,
    "scaled_solutions": ArchiveStrategyRegistration,
    "sinusoidal_field": MatrixStrategyRegistration,
    "jacobi_probes": MatrixStrategyRegistration,
    "solution_archive": ArchiveStrategyRegistration,
    "sparse_rhs": MatrixStrategyRegistration,
    "uniform_forward": MatrixStrategyRegistration,
    "uniform_inverse": MatrixStrategyRegistration,
    "validated_archive": ArchiveStrategyRegistration,
}


def test_all_strategy_modules_register_on_package_import() -> None:
    """Every strategy is registered, with the kind of input it consumes, on package import."""
    assert {
        name: type(registration) for name, registration in _registry._strategies.items()
    } == EXPECTED_STRATEGIES


def test_gaussian_inverse_runs_after_package_import(
    spd_matrix: np.ndarray,
) -> None:
    """Gaussian inverse is reachable through the public dispatcher."""
    sample_count = 3
    result = solved_rows(
        run_generation(
            "gaussian_inverse",
            spd_matrix,
            cfg={"samples": sample_count, "seed": 0, "mu": 0.0, "sigma": 1.0},
            archive_source=ARCHIVE_SOURCE,
        )
    )
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.shape == (sample_count, spd_matrix.shape[0])
    assert result.solutions.shape == (sample_count, spd_matrix.shape[0])
