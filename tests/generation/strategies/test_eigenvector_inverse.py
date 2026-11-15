"""Tests for EigenvectorInverseStrategy (eigenvector_inverse): solver choice and `which`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from neuralls.domain.generation import run_generation
from tests.generation_support import solved_rows


@pytest.mark.parametrize(
    ("options", "tolerance"),
    [
        pytest.param({"which": "largest"}, 1e-10, id="which-largest"),
        pytest.param(
            {"solve_config": {"method": "cg", "rtol": 1e-10, "atol": 1e-12, "max_iters": 500}},
            1e-8,
            id="cg-solve",
        ),
    ],
)
def test_eigenvector_inverse_solves_ax_equals_b(
    spd_matrix: np.ndarray, options: dict[str, Any], tolerance: float
) -> None:
    result = solved_rows(
        run_generation("eigenvector_inverse", spd_matrix, cfg={"samples": 3, "seed": 0, **options})
    )

    assert result.rhs is not None
    assert result.solutions is not None
    np.testing.assert_allclose(
        spd_matrix @ result.solutions.T, result.rhs.T, rtol=tolerance, atol=tolerance
    )
