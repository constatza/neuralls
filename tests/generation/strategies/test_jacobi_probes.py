"""Tests for JacobiProbesStrategy."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from neuralls.domain.generation import run_generation
from tests.generation_support import residual_trace_rows


@pytest.mark.parametrize("samples", [1, 4, 5, 8])
def test_jacobi_probes_emit_one_trace_row_per_sample(spd_matrix: np.ndarray, samples: int) -> None:
    cfg = {"samples": samples, "seed": 0, "stop": 5, "start": 1}

    result = residual_trace_rows(run_generation("jacobi_probes", spd_matrix, cfg=cfg))

    traces = result.residual_traces
    assert traces is not None
    n = spd_matrix.shape[0]
    assert traces.residuals.shape == (samples, n)
    assert traces.solutions.shape == (samples, n)


@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param({"samples": 5, "seed": 42, "stop": 5, "start": 1}, id="single-step"),
        pytest.param({"samples": 3, "seed": 1, "stop": 8, "start": 5}, id="multi-step"),
    ],
)
def test_jacobi_probes_rhs_is_a_times_solution(spd_matrix: np.ndarray, cfg: dict[str, int]) -> None:
    result = residual_trace_rows(run_generation("jacobi_probes", spd_matrix, cfg=cfg))

    traces = result.residual_traces
    assert traces is not None
    np.testing.assert_allclose(traces.residuals, traces.solutions @ spd_matrix.T, rtol=1e-10)


def test_jacobi_probes_deterministic(spd_matrix: np.ndarray) -> None:
    """Same seed produces identical output across two calls."""
    cfg = {"samples": 4, "seed": 7, "stop": 5, "start": 1}

    result1 = residual_trace_rows(run_generation("jacobi_probes", spd_matrix, cfg=cfg))
    result2 = residual_trace_rows(run_generation("jacobi_probes", spd_matrix, cfg=cfg))

    assert result1.residual_traces is not None and result2.residual_traces is not None
    np.testing.assert_array_equal(
        result1.residual_traces.residuals, result2.residual_traces.residuals
    )
    np.testing.assert_array_equal(
        result1.residual_traces.solutions, result2.residual_traces.solutions
    )


def test_jacobi_probes_rademacher_distribution(spd_matrix: np.ndarray) -> None:
    """The 'rademacher' probe distribution runs successfully and produces valid shapes."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 3, "seed": 0, "stop": 5, "start": 1, "probe_distribution": "rademacher"}

    result = residual_trace_rows(run_generation("jacobi_probes", spd_matrix, cfg=cfg))
    assert result.residual_traces is not None

    assert result.residual_traces.solutions.shape == (3, n)


def test_jacobi_probes_samples_minus_one_rejected(spd_matrix: np.ndarray) -> None:
    """samples=-1 is rejected at config construction (SOLID finding 2: no archive source exists)."""
    with pytest.raises(ValidationError):
        residual_trace_rows(
            run_generation(
                "jacobi_probes",
                spd_matrix,
                cfg={"samples": -1, "seed": 0, "stop": 5, "start": 1},
            )
        )


# ---------------------------------------------------------------------------
# Multi-step harvesting (a genuinely new capability)
# ---------------------------------------------------------------------------


def test_jacobi_probes_multi_step_keeps_consistent_2d_shape(
    spd_matrix: np.ndarray,
) -> None:
    """`start` set to harvest 3 sweep depths per probe: output stays a 2D trace block.

    Guards against the output type changing shape/kind (flat 1D vs 2D)
    depending on how many steps are kept — it must always be a 2D block of
    (sample_indices, iteration_indices)-paired rows. `samples` here is a
    flattened-row budget (same convention as residuals.py),
    not a probe count — 2 base probes x 3 kept rows each = 6, exactly.
    """
    stop, rows_per_probe, num_probes = 10, 3, 2
    samples = rows_per_probe * num_probes
    cfg = {"samples": samples, "seed": 0, "stop": stop, "start": stop - 2}

    result = residual_trace_rows(run_generation("jacobi_probes", spd_matrix, cfg=cfg))
    traces = result.residual_traces
    assert traces is not None
    assert traces.solutions.ndim == 2
    assert traces.solutions.shape[0] == samples
    assert traces.solutions.shape[1] == spd_matrix.shape[0]
    np.testing.assert_array_equal(
        traces.iteration_indices[:3], np.array([stop - 2, stop - 1, stop], dtype=np.int64)
    )
