"""CG-error snapshots: Gaussian probes propagated by a fixed number of CG iterations.

Each probe ``z ~ N(0, I)`` is the known solution of ``A x = b`` with ``b = A z``. CG runs
exactly ``window.stop`` iterations from ``x_0 = 0``, and the kept rows are the pairs
``(r_k, e_k = z - x_k)`` of the iterations ``window`` selects, optionally scaled per row
(``snapshot_scaling``). These rows are the POD-CG snapshot source.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neuralls.domain.normalization import ResidualTraceSamples
from neuralls.shared.types import SystemMatrix

from ..counts import resolve_trace_generation_counts, trace_rows_per_base_system
from ..interfaces import ArchiveData, ResidualTraceRows, TracingSolverCallable
from ..ports import ArchiveSource
from ..runner import register_single_rhs_strategy
from ..strategy_configs import CgErrorsConfig
from ..trace_utils import _trim_error_traces
from ..trajectory import cg_error_traces, draw_probes, scale_rows
from ..transforms import ComputeRhsTransform


@register_single_rhs_strategy(rows_per_base_system=trace_rows_per_base_system(CgErrorsConfig))
class CgErrorsStrategy:
    """Generate ``(r_k, e_k)`` snapshot rows from CG runs on Gaussian probes."""

    name = "cg_errors"
    ConfigType = CgErrorsConfig

    def generate(
        self,
        matrix: SystemMatrix,
        *,
        cfg: dict[str, Any],
        solver: TracingSolverCallable,
        archive_source: ArchiveSource,
        single_rhs: np.ndarray | None = None,
        archive: ArchiveData | None = None,
    ) -> ResidualTraceRows:
        """Generate CG-error snapshot rows.

        Registered as a single-RHS strategy only to receive the tracing solver: the probes
        are its own systems, so ``archive_source``, ``single_rhs`` and ``archive`` are
        ignored.

        Args:
            matrix: System matrix (already scaled).
            cfg: Configuration dictionary, validated against ``CgErrorsConfig``.
            solver: Tracing CG solver.
            archive_source: Unused.
            single_rhs: Unused.
            archive: Unused.

        Returns:
            ResidualTraceRows: ``residuals`` holds ``r_k`` and ``solutions`` holds ``e_k``,
                one row per kept iteration per probe.
        """
        del archive_source, single_rhs, archive
        config = CgErrorsConfig(**cfg)
        window = config.window
        num_base_systems, final_rows = resolve_trace_generation_counts(
            config.samples, window=window, available_systems=None, strategy_name=self.name
        )
        rng = np.random.default_rng(config.seed)
        probes = draw_probes(rng, num_base_systems, matrix.shape[0], "gaussian")
        rhs = ComputeRhsTransform(matrix).transform(probes)
        traces = _trim_error_traces(
            cg_error_traces(matrix, rhs, probes, window, solver), final_rows
        )
        residuals, errors = scale_rows(traces.residuals, traces.errors, config.snapshot_scaling)
        return ResidualTraceRows(
            matrix=matrix,
            residual_traces=ResidualTraceSamples(
                residuals=residuals,
                solutions=errors,
                sample_indices=traces.sample_indices,
                iteration_indices=traces.iteration_indices,
            ),
        )
