"""Tests for SmootherFilteredProbesStrategy."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchalg.preconditioners.implementations.pod import apply_jacobi_damping

from neuralls.domain.generation import run_generation


def test_smoother_filtered_probes_registered() -> None:
    """SmootherFilteredProbesStrategy is registered under 'smoother_filtered_probes'."""
    from neuralls.domain.generation.runner import _registry

    assert "smoother_filtered_probes" in _registry._strategies


def test_smoother_filtered_probes_shapes(spd_matrix: np.ndarray) -> None:
    """Output is a residual_traces block, 2D, one row per probe by default."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 4, "seed": 0, "stop": 5}

    result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)

    assert result.residual_traces is not None
    assert result.residual_traces.residuals.shape == (4, n)
    assert result.residual_traces.solutions.shape == (4, n)


def test_smoother_filtered_probes_sample_count(spd_matrix: np.ndarray) -> None:
    """Different sample counts produce correct output sizes."""
    for count in (1, 5, 8):
        cfg = {"samples": count, "seed": 0, "stop": 5}
        result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)
        assert result.residual_traces is not None
        assert result.residual_traces.solutions.shape[0] == count


def test_smoother_filtered_probes_computes_rhs(spd_matrix: np.ndarray) -> None:
    """RHS = A @ x for all generated samples."""
    cfg = {"samples": 5, "seed": 42, "stop": 5}
    result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)

    traces = result.residual_traces
    assert traces is not None
    for i in range(traces.solutions.shape[0]):
        expected = spd_matrix @ traces.solutions[i]
        np.testing.assert_allclose(traces.residuals[i], expected, rtol=1e-10)


def test_smoother_filtered_probes_deterministic(spd_matrix: np.ndarray) -> None:
    """Same seed produces identical output across two calls."""
    cfg = {"samples": 4, "seed": 7, "stop": 5}

    result1 = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)
    result2 = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)

    assert result1.residual_traces is not None and result2.residual_traces is not None
    np.testing.assert_array_equal(
        result1.residual_traces.residuals, result2.residual_traces.residuals
    )
    np.testing.assert_array_equal(
        result1.residual_traces.solutions, result2.residual_traces.solutions
    )


def test_smoother_filtered_probes_rademacher_distribution(spd_matrix: np.ndarray) -> None:
    """The 'rademacher' probe distribution runs successfully and produces valid shapes."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 3, "seed": 0, "stop": 5, "probe_distribution": "rademacher"}

    result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)

    assert result.residual_traces is not None
    assert result.residual_traces.solutions.shape == (3, n)


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
    cfg_few = {"samples": 20, "seed": 0, "stop": 1}
    cfg_many = {"samples": 20, "seed": 0, "stop": 20}

    few_steps = run_generation("smoother_filtered_probes", poisson_1d, cfg=cfg_few)
    many_steps = run_generation("smoother_filtered_probes", poisson_1d, cfg=cfg_many)

    assert few_steps.residual_traces is not None and many_steps.residual_traces is not None
    few_norms = np.linalg.norm(few_steps.residual_traces.solutions, axis=1)
    many_norms = np.linalg.norm(many_steps.residual_traces.solutions, axis=1)
    assert np.mean(many_norms) < np.mean(few_norms)


def test_smoother_filtered_probes_requires_stop(spd_matrix: np.ndarray) -> None:
    """Omitting the required 'stop' field must raise a validation error, not run with a silent default."""
    cfg = {"samples": 3, "seed": 0}

    with pytest.raises(Exception, match="stop"):
        run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)


def test_smoother_filtered_probes_samples_minus_one_rejected(spd_matrix: np.ndarray) -> None:
    """samples=-1 is rejected at config construction (SOLID finding 2: no archive source exists)."""
    with pytest.raises(Exception, match="samples"):
        run_generation(
            "smoother_filtered_probes", spd_matrix, cfg={"samples": -1, "seed": 0, "stop": 5}
        )


# ---------------------------------------------------------------------------
# Default (last-only) matches the old, final-only apply_jacobi_damping call exactly
# ---------------------------------------------------------------------------


def test_smoother_filtered_probes_default_matches_apply_jacobi_damping(
    spd_matrix: np.ndarray,
) -> None:
    """With `start` unset, output matches a direct apply_jacobi_damping call bit-for-bit."""
    omega, stop, samples = 0.67, 6, 5
    cfg = {"samples": samples, "seed": 3, "stop": stop, "omega": omega}

    result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)
    traces = result.residual_traces
    assert traces is not None
    assert traces.solutions.shape == (samples, spd_matrix.shape[0])

    # Reproduce the same probes independently and damp them directly.
    rng = np.random.default_rng(3)
    probes = rng.standard_normal((samples, spd_matrix.shape[0])).astype(spd_matrix.dtype)
    expected = apply_jacobi_damping(
        torch.as_tensor(probes), torch.as_tensor(spd_matrix), omega=omega, steps=stop
    ).numpy()

    np.testing.assert_allclose(traces.solutions, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(traces.iteration_indices, np.full(samples, stop, dtype=np.int64))


# ---------------------------------------------------------------------------
# Multi-step harvesting (a genuinely new capability)
# ---------------------------------------------------------------------------


def test_smoother_filtered_probes_multi_step_keeps_consistent_2d_shape(
    spd_matrix: np.ndarray,
) -> None:
    """`start` set to harvest 3 sweep depths per probe: output stays a 2D trace block.

    Guards against the output type changing shape/kind (flat 1D vs 2D)
    depending on how many steps are kept — it must always be a 2D block of
    (sample_indices, iteration_indices)-paired rows. `samples` here is a
    flattened-row budget (same convention as residuals.py/search_directions.py),
    not a probe count — 2 base probes x 3 kept rows each = 6, exactly.
    """
    stop, rows_per_probe, num_probes = 10, 3, 2
    samples = rows_per_probe * num_probes
    cfg = {"samples": samples, "seed": 0, "stop": stop, "start": stop - 2}

    result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)
    traces = result.residual_traces
    assert traces is not None
    assert traces.solutions.ndim == 2
    assert traces.solutions.shape[0] == samples
    assert traces.solutions.shape[1] == spd_matrix.shape[0]
    np.testing.assert_array_equal(
        traces.iteration_indices[:3], np.array([stop - 2, stop - 1, stop], dtype=np.int64)
    )


def test_smoother_filtered_probes_multi_step_rhs_consistent(spd_matrix: np.ndarray) -> None:
    """RHS = A @ x still holds per-row when multiple sweep depths are kept per probe."""
    cfg = {"samples": 3, "seed": 1, "stop": 8, "start": 5}

    result = run_generation("smoother_filtered_probes", spd_matrix, cfg=cfg)
    traces = result.residual_traces
    assert traces is not None
    for i in range(traces.solutions.shape[0]):
        expected = spd_matrix @ traces.solutions[i]
        np.testing.assert_allclose(traces.residuals[i], expected, rtol=1e-10)
