"""Tests for EigenvectorForwardStrategy (eigenvector_forward).

Forward mode: generates solutions as eigenvector combinations, then computes b = A @ x.
"""

from __future__ import annotations

import numpy as np

from neuralls.domain.generation import run_generation


def test_eigenvector_forward_registered() -> None:
    """EigenvectorForwardStrategy is registered under 'eigenvector_forward'."""
    from neuralls.domain.generation.runner import _registry

    assert "eigenvector_forward" in _registry._strategies


def test_eigenvector_forward_shapes(spd_matrix: np.ndarray) -> None:
    """Solutions and RHS have correct shapes."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 4, "seed": 0}

    result = run_generation("eigenvector_forward", spd_matrix, cfg=cfg)
    assert result.rhs is not None
    assert result.solutions is not None

    assert result.solutions.shape == (4, n)
    assert result.rhs.shape == (4, n)


def test_eigenvector_forward_sample_count(spd_matrix: np.ndarray) -> None:
    """Different sample counts produce correct output sizes."""
    for count in (1, 3, 6):
        result = run_generation(
            "eigenvector_forward", spd_matrix, cfg={"samples": count, "seed": 0}
        )
        assert result.rhs is not None
        assert result.rhs.shape[0] == count


def test_eigenvector_forward_rhs_equals_ax(spd_matrix: np.ndarray) -> None:
    """RHS = A @ x for every generated sample."""
    result = run_generation("eigenvector_forward", spd_matrix, cfg={"samples": 5, "seed": 42})
    assert result.solutions is not None
    assert result.rhs is not None

    assert result.rhs.ndim == 2 and result.solutions.ndim == 2
    for i in range(result.solutions.shape[0]):
        assert result.rhs is not None
        assert result.solutions is not None
        np.testing.assert_allclose(result.rhs[i], spd_matrix @ result.solutions[i], rtol=1e-12)


def test_eigenvector_forward_deterministic(spd_matrix: np.ndarray) -> None:
    """Same seed yields identical output on repeated calls."""
    cfg = {"samples": 4, "seed": 7}

    r1 = run_generation("eigenvector_forward", spd_matrix, cfg=cfg)
    assert r1.solutions is not None
    r2 = run_generation("eigenvector_forward", spd_matrix, cfg=cfg)
    assert r2.solutions is not None

    assert r1.solutions.ndim == 2 and r2.solutions.ndim == 2
    np.testing.assert_array_equal(r1.solutions, r2.solutions)
    np.testing.assert_array_equal(r1.rhs, r2.rhs)


def test_eigenvector_forward_different_seeds(spd_matrix: np.ndarray) -> None:
    """Different seeds produce different output."""
    r1 = run_generation("eigenvector_forward", spd_matrix, cfg={"samples": 4, "seed": 1})
    assert r1.solutions is not None
    r2 = run_generation("eigenvector_forward", spd_matrix, cfg={"samples": 4, "seed": 2})
    assert r2.solutions is not None

    assert r1.solutions.ndim == 2 and r2.solutions.ndim == 2
    assert not np.array_equal(r1.solutions, r2.solutions)


def test_eigenvector_forward_which_smallest(spd_matrix: np.ndarray) -> None:
    """which='smallest' runs without error and produces correct shapes."""
    n = spd_matrix.shape[0]
    result = run_generation(
        "eigenvector_forward", spd_matrix, cfg={"samples": 3, "seed": 0, "which": "smallest"}
    )
    assert result.rhs is not None
    assert result.rhs.shape == (3, n)


def test_eigenvector_forward_which_largest(spd_matrix: np.ndarray) -> None:
    """which='largest' runs without error and produces correct shapes."""
    n = spd_matrix.shape[0]
    result = run_generation(
        "eigenvector_forward", spd_matrix, cfg={"samples": 3, "seed": 0, "which": "largest"}
    )
    assert result.rhs is not None
    assert result.rhs.shape == (3, n)


def test_eigenvector_forward_no_traces(spd_matrix: np.ndarray) -> None:
    """Forward strategy produces no residual or error traces."""
    result = run_generation("eigenvector_forward", spd_matrix, cfg={"samples": 2, "seed": 0})

    assert result.residual_traces is None
    assert result.error_traces is None
