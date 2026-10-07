"""Tests for EigenvectorInverseStrategy (eigenvector_inverse).

Inverse mode: generates RHS as eigenvector combinations, then solves x = A^-1 @ b.
"""

from __future__ import annotations

import numpy as np

from neuralls.domain.generation import run_generation


def test_eigenvector_inverse_registered() -> None:
    """EigenvectorInverseStrategy is registered under 'eigenvector_inverse'."""
    from neuralls.domain.generation.runner import _registry

    assert "eigenvector_inverse" in _registry._strategies


def test_eigenvector_inverse_shapes(spd_matrix: np.ndarray) -> None:
    """Solutions and RHS have correct shapes."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 4, "seed": 0}

    result = run_generation("eigenvector_inverse", spd_matrix, cfg=cfg)
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.shape == (4, n)
    assert result.solutions.shape == (4, n)


def test_eigenvector_inverse_sample_count(spd_matrix: np.ndarray) -> None:
    """Different sample counts produce correct output sizes."""
    for count in (1, 3, 5):
        result = run_generation(
            "eigenvector_inverse", spd_matrix, cfg={"samples": count, "seed": 0}
        )
        assert result.solutions is not None
        assert result.solutions.shape[0] == count


def test_eigenvector_inverse_solves_ax_equals_b(spd_matrix: np.ndarray) -> None:
    """Returned solutions satisfy A @ x = b within tolerance (direct solve)."""
    result = run_generation("eigenvector_inverse", spd_matrix, cfg={"samples": 4, "seed": 42})
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.ndim == 2 and result.solutions.ndim == 2
    for i in range(result.solutions.shape[0]):
        np.testing.assert_allclose(
            spd_matrix @ result.solutions[i], result.rhs[i], rtol=1e-10, atol=1e-10
        )


def test_eigenvector_inverse_deterministic(spd_matrix: np.ndarray) -> None:
    """Same seed yields identical output on repeated calls."""
    cfg = {"samples": 4, "seed": 7}

    r1 = run_generation("eigenvector_inverse", spd_matrix, cfg=cfg)
    assert r1.solutions is not None
    assert r1.rhs is not None
    r2 = run_generation("eigenvector_inverse", spd_matrix, cfg=cfg)
    assert r2.solutions is not None
    assert r2.rhs is not None

    assert r1.rhs.ndim == 2 and r2.rhs.ndim == 2
    np.testing.assert_array_equal(r1.rhs, r2.rhs)
    assert r1.solutions.ndim == 2 and r2.solutions.ndim == 2
    np.testing.assert_array_equal(r1.solutions, r2.solutions)


def test_eigenvector_inverse_different_seeds(spd_matrix: np.ndarray) -> None:
    """Different seeds produce different output."""
    r1 = run_generation("eigenvector_inverse", spd_matrix, cfg={"samples": 4, "seed": 1})
    assert r1.rhs is not None
    r2 = run_generation("eigenvector_inverse", spd_matrix, cfg={"samples": 4, "seed": 2})
    assert r2.rhs is not None

    assert r1.rhs.ndim == 2 and r2.rhs.ndim == 2
    assert not np.array_equal(r1.rhs, r2.rhs)


def test_eigenvector_inverse_cg_solve(spd_matrix: np.ndarray) -> None:
    """CG solve method also satisfies A @ x = b within tolerance."""
    cfg = {
        "samples": 3,
        "seed": 0,
        "solve_config": {"method": "cg", "rtol": 1e-10, "atol": 1e-12, "max_iters": 500},
    }

    result = run_generation("eigenvector_inverse", spd_matrix, cfg=cfg)
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.ndim == 2 and result.solutions.ndim == 2
    for i in range(result.solutions.shape[0]):
        np.testing.assert_allclose(
            spd_matrix @ result.solutions[i], result.rhs[i], rtol=1e-8, atol=1e-8
        )


def test_eigenvector_inverse_which_largest(spd_matrix: np.ndarray) -> None:
    """which='largest' runs without error and satisfies A @ x = b."""
    result = run_generation(
        "eigenvector_inverse", spd_matrix, cfg={"samples": 3, "seed": 0, "which": "largest"}
    )
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.ndim == 2 and result.solutions.ndim == 2
    for i in range(result.solutions.shape[0]):
        np.testing.assert_allclose(
            spd_matrix @ result.solutions[i], result.rhs[i], rtol=1e-10, atol=1e-10
        )


def test_eigenvector_inverse_no_traces(spd_matrix: np.ndarray) -> None:
    """Inverse strategy produces no residual or error traces."""
    result = run_generation("eigenvector_inverse", spd_matrix, cfg={"samples": 2, "seed": 0})

    assert result.residual_traces is None
    assert result.error_traces is None
