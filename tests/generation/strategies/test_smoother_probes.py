"""Tests for SmootherFilteredProbesStrategy."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation import run_generation


def test_smoother_filtered_probes_registered() -> None:
    """SmootherFilteredProbesStrategy is registered under 'smoother_filtered_probes'."""
    from neuralls.domain.generation.runner import _registry

    assert "smoother_filtered_probes" in _registry._strategies


def test_smoother_filtered_probes_shapes(spd_matrix: np.ndarray) -> None:
    """Output arrays have correct shapes."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 4, "seed": 0, "steps": 5}

    result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)

    assert result.rhs is not None
    assert result.rhs.shape == (4, n)
    assert result.solutions is not None
    assert result.solutions.shape == (4, n)


def test_smoother_filtered_probes_sample_count(spd_matrix: np.ndarray) -> None:
    """Different sample counts produce correct output sizes."""
    for count in (1, 5, 8):
        cfg = {"samples": count, "seed": 0, "steps": 5}
        result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)
        assert result.rhs is not None
        assert result.rhs.shape[0] == count


def test_smoother_filtered_probes_computes_rhs(spd_matrix: np.ndarray) -> None:
    """RHS = A @ x for all generated samples."""
    cfg = {"samples": 5, "seed": 42, "steps": 5}
    result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)

    assert result.rhs is not None
    assert result.solutions is not None
    for i in range(result.solutions.shape[0]):
        expected = spd_matrix @ result.solutions[i]
        np.testing.assert_allclose(result.rhs[i], expected, rtol=1e-10)


def test_smoother_filtered_probes_deterministic(spd_matrix: np.ndarray) -> None:
    """Same seed produces identical output across two calls."""
    cfg = {"samples": 4, "seed": 7, "steps": 5}

    result1 = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)
    result2 = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)

    assert result1.rhs is not None and result2.rhs is not None
    np.testing.assert_array_equal(result1.rhs, result2.rhs)
    assert result1.solutions is not None and result2.solutions is not None
    np.testing.assert_array_equal(result1.solutions, result2.solutions)


def test_smoother_filtered_probes_rademacher_distribution(spd_matrix: np.ndarray) -> None:
    """The 'rademacher' probe distribution runs successfully and produces valid shapes."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 3, "seed": 0, "steps": 5, "probe_distribution": "rademacher"}

    result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)

    assert result.solutions is not None
    assert result.solutions.shape == (3, n)


def test_smoother_filtered_probes_more_steps_damps_more() -> None:
    """More Jacobi damping steps must not increase a probe's norm (damping only removes energy).

    Weighted-Jacobi with `omega=0.67` is only a contraction
    (`rho(I - omega D^-1 A) < 1`) for matrices in its intended regime — the
    default's own docstring says "isotropic SPD problems where
    rho(D^-1 A) ~= 2", e.g. FEM/Poisson-type stiffness matrices, not an
    arbitrary dense SPD matrix (`spd_matrix`'s `A^T A + I` construction gives
    no diagonal-dominance guarantee at all, so damping can diverge there).
    A small 1D-Poisson-style tridiagonal matrix is the standard well-behaved
    case for this claim.
    """
    n = 10
    poisson_1d = 2 * np.eye(n) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
    cfg_few = {"samples": 20, "seed": 0, "steps": 1}
    cfg_many = {"samples": 20, "seed": 0, "steps": 20}

    few_steps = run_generation("smoother_filtered_probes", poisson_1d, cfg=cfg_few)
    many_steps = run_generation("smoother_filtered_probes", poisson_1d, cfg=cfg_many)

    assert few_steps.solutions is not None and many_steps.solutions is not None
    few_norms = np.linalg.norm(few_steps.solutions, axis=1)
    many_norms = np.linalg.norm(many_steps.solutions, axis=1)
    assert np.mean(many_norms) < np.mean(few_norms)


def test_smoother_filtered_probes_requires_steps(spd_matrix: np.ndarray) -> None:
    """Omitting the required 'steps' field must raise a validation error, not run with a silent default."""
    cfg = {"samples": 3, "seed": 0}

    with pytest.raises(Exception, match="steps"):
        run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)
