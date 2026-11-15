"""Tests for EigenvectorForwardStrategy (eigenvector_forward): the `which` selection."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation import run_generation
from tests.generation_support import solved_rows


@pytest.mark.parametrize("which", ["smallest", "largest"])
def test_eigenvector_forward_which_selection_gives_consistent_rows(
    spd_matrix: np.ndarray, which: str
) -> None:
    result = solved_rows(
        run_generation(
            "eigenvector_forward", spd_matrix, cfg={"samples": 3, "seed": 0, "which": which}
        )
    )

    assert result.rhs is not None
    assert result.solutions is not None
    assert result.rhs.shape == (3, spd_matrix.shape[0])
    np.testing.assert_allclose(result.rhs, result.solutions @ spd_matrix.T, rtol=1e-12)
