"""The eigenvector inverse strategy solves with the configured linear solver."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from neuralls.domain.generation.strategies.eigenvector import EigenvectorInverseStrategy

SAMPLES = 5


@pytest.fixture
def spd_matrix() -> np.ndarray:
    """A seeded, well-conditioned symmetric positive-definite matrix."""
    n = 20
    A = np.random.default_rng(2).standard_normal((n, n))
    return A.T @ A + 5 * np.eye(n)


@pytest.mark.parametrize(
    ("solve_config", "tolerance"),
    [
        pytest.param({"method": "direct", "assume_pos_def": True}, 1e-10, id="direct"),
        pytest.param({"method": "cg", "rtol": 1e-12, "atol": 0.0, "max_iters": 100}, 1e-8, id="cg"),
    ],
)
def test_eigenvector_inverse_solves_with_the_configured_method(
    spd_matrix: np.ndarray, solve_config: dict[str, Any], tolerance: float
) -> None:
    samples = EigenvectorInverseStrategy().generate(
        spd_matrix,
        cfg={
            "samples": SAMPLES,
            "which": "smallest",
            "num_eigenvectors": 5,
            "include_eigenvectors": True,
            "seed": 42,
            "solve_config": solve_config,
        },
    )

    assert samples.solutions is not None
    assert samples.rhs is not None
    assert samples.rhs.shape == (SAMPLES, spd_matrix.shape[0])
    assert samples.solutions.shape == samples.rhs.shape
    relative = np.linalg.norm(samples.rhs - samples.solutions @ spd_matrix.T, axis=1) / (
        np.linalg.norm(samples.rhs, axis=1)
    )
    assert np.all(relative < tolerance)
