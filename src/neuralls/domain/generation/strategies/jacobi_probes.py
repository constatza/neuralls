"""Jacobi-filtered probe snapshots: random vectors damped by weighted-Jacobi sweeps.

Instead of deriving "algebraically smooth" error snapshots from CG trajectories
(``cg_errors.py``), this strategy synthesizes them directly: random probe vectors are
passed through ``window.stop`` weighted-Jacobi damping sweeps, and ``window`` selects
which of those sweeps are kept (by default just the final, fully-damped one). A Jacobi
smoother's error-propagation operator G = I - omega D^-1 A quickly attenuates any
direction it handles well, so what survives after ``window.stop`` sweeps is, by
construction, smoother-resistant — exactly the directions a multigrid coarse-grid
correction needs to cover. These rows are the POD-JP snapshot source.

The damping sweep itself is torchalg's (``apply_jacobi_damping_trajectory``, or its
sparse ``JacobiSmoother`` for CSR), so "algebraically smooth" means the same weighted-
Jacobi map the AMG smoother runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neuralls.domain.normalization import ResidualTraceSamples
from neuralls.shared.types import SystemMatrix

from ..counts import resolve_trace_generation_counts, trace_rows_per_base_system
from ..interfaces import ArchiveData, ResidualTraceRows
from ..runner import register_strategy
from ..strategy_configs import JacobiProbesConfig
from ..trace_utils import _trim_residual_traces
from ..trajectory import draw_probes, jacobi_probe_traces, scale_rows


@register_strategy(rows_per_base_system=trace_rows_per_base_system(JacobiProbesConfig))
class JacobiProbesStrategy:
    """Generate ``(A e_k, e_k)`` snapshot rows by Jacobi-damping random probes."""

    name = "jacobi_probes"
    ConfigType = JacobiProbesConfig

    def generate(
        self,
        matrix: SystemMatrix,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> ResidualTraceRows:
        """Generate Jacobi-filtered probe snapshot rows.

        Args:
            matrix: System matrix (already scaled).
            cfg: Configuration dictionary, validated against ``JacobiProbesConfig``.
            archive: Unused: the probes are drawn, never read.

        Returns:
            ResidualTraceRows: ``solutions`` holds the damped probes ``e_k`` and
                ``residuals`` holds ``A e_k``, one row per kept sweep per probe.
        """
        del archive
        config = JacobiProbesConfig(**cfg)
        window = config.window
        num_base_probes, final_rows = resolve_trace_generation_counts(
            config.samples, window=window, available_systems=None, strategy_name=self.name
        )
        rng = np.random.default_rng(config.seed)
        probes = draw_probes(rng, num_base_probes, matrix.shape[0], config.probe_distribution)
        probes = probes.astype(matrix.dtype, copy=False)
        traces = _trim_residual_traces(
            jacobi_probe_traces(matrix, probes, window, config.omega), final_rows
        )
        residuals, errors = scale_rows(traces.residuals, traces.solutions, config.snapshot_scaling)
        return ResidualTraceRows(
            matrix=matrix,
            residual_traces=ResidualTraceSamples(
                residuals=residuals,
                solutions=errors,
                sample_indices=traces.sample_indices,
                iteration_indices=traces.iteration_indices,
            ),
        )
