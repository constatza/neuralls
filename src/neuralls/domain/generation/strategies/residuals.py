"""Archive-backed residual trace strategy with error targets.

True solutions come from archive files (or the explicit solution rows), their RHS is
``b = A x`` (or the archived RHS), and CG runs from ``x_0 = 0`` through the shared
``trajectory.cg_error_traces`` propagation, keeping the residuals together with the
exact error targets ``e_k = x_true - x_k`` of the selected iterations. These rows are
neural-network training data; the random-probe POD snapshot source is ``cg_errors``.
"""

from __future__ import annotations

import numpy as np

from ..counts import resolve_trace_generation_counts, trace_rows_per_base_system
from ..interfaces import ArchiveData, ArchiveField, ErrorTraceRows, TracingSolverCallable
from ..matrix_operator import MatrixOperator
from ..ports import ArchiveSource
from ..providers import HybridInputProvider, provide_solutions
from ..runner import register_single_rhs_strategy
from ..strategy_configs import ResidualErrorConfig
from ..trace_utils import _referenced_sample_count, _trim_error_traces
from ..trajectory import cg_error_traces
from ..transforms import ComputeRhsTransform


@register_single_rhs_strategy(
    rows_per_base_system=trace_rows_per_base_system(ResidualErrorConfig),
)
class ResidualsStrategy:
    ConfigType = ResidualErrorConfig
    name = "residuals"

    def _resolve_available_systems(
        self,
        matrix: np.ndarray,
        config: ResidualErrorConfig,
        archive_source: ArchiveSource,
        archive: ArchiveData | None,
        rng: np.random.Generator,
    ) -> int | None:
        if config.solutions_glob is not None:
            return len(
                provide_solutions(
                    matrix,
                    -1,
                    rng,
                    archive_source=archive_source,
                    solutions_glob=config.solutions_glob,
                    archive=archive,
                    shuffle=config.shuffle,
                    seed=config.seed,
                    strategy_name=self.name,
                )
            )
        if archive is not None and archive.lhs is not None:
            return int(archive.lhs.shape[0])
        return None

    def _provide_true_solutions(
        self,
        matrix: np.ndarray,
        count: int,
        config: ResidualErrorConfig,
        archive_source: ArchiveSource,
        archive: ArchiveData | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return provide_solutions(
            matrix,
            count,
            rng,
            archive_source=archive_source,
            solutions_glob=config.solutions_glob,
            archive=archive,
            shuffle=config.shuffle,
            seed=config.seed,
            strategy_name=self.name,
            file_indices=config.file_indices,
        )

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict,
        solver: TracingSolverCallable,
        archive_source: ArchiveSource,
        single_rhs: np.ndarray | None = None,
        archive: ArchiveData | None = None,
    ) -> ErrorTraceRows:
        """Generate samples with full residual error traces.

        Supports two modes:
        - Single RHS mode: If single_rhs provided, run CG multiple times on the same RHS
        - Multiple RHS mode: If single_rhs is None, generate N different RHS vectors

        Args:
            matrix: System matrix
            cfg: Configuration dictionary
            archive_source: Lists and loads the files ``solutions_glob`` matches
            single_rhs: Optional single RHS vector. If provided, all samples solve A @ x = single_rhs
            archive: Optional archive data to seed generation

        Returns:
            GeneratedSamples with error_traces populated
        """
        # Validate and convert to typed config
        config = ResidualErrorConfig(**cfg)

        window = config.window
        rng = np.random.default_rng(config.seed)
        available_systems: int | None = None

        if single_rhs is None and config.samples == -1:
            available_systems = self._resolve_available_systems(
                matrix, config, archive_source, archive, rng
            )

        num_base_systems, final_rows = resolve_trace_generation_counts(
            config.samples,
            window=window,
            available_systems=available_systems,
            strategy_name=self.name,
        )

        # Choose mode based on single_rhs parameter
        if single_rhs is not None:
            # Mode 1: Single RHS - run CG multiple times on the SAME RHS
            # Solve exactly to get "true" solution for error target computation
            # LU (not Cholesky): the general solve this replaced made no SPD assumption.
            true_sol = MatrixOperator(matrix).solve_direct(single_rhs, assume_pos_def=False)

            # Create array of identical RHS and solution vectors
            rhs_samples = np.tile(single_rhs, (num_base_systems, 1))
            sols = np.tile(true_sol, (num_base_systems, 1))
        else:
            sols = self._provide_true_solutions(
                matrix,
                num_base_systems,
                config,
                archive_source,
                archive,
                rng,
            )

            # Layer 2: Transform (compute RHS or load from archive)
            rhs_provider = HybridInputProvider(archive=archive, field=ArchiveField.RHS, scale=1.0)
            rhs_from_archive = (
                archive is not None
                and archive.rhs is not None
                and archive.rhs.shape[0] >= num_base_systems
            )

            if rhs_from_archive:
                # Use RHS directly from archive
                rhs_samples = rhs_provider.provide(matrix, count=num_base_systems, rng=rng)
            else:
                # Compute RHS = A @ x
                transform = ComputeRhsTransform(matrix)
                rhs_samples = transform.transform(sols)

        error_traces = cg_error_traces(matrix, rhs_samples, sols, window, solver)
        error_traces = _trim_error_traces(error_traces, final_rows)
        referenced_samples = _referenced_sample_count(error_traces.sample_indices)

        return ErrorTraceRows(
            matrix=matrix,
            rhs=rhs_samples[:referenced_samples],
            solutions=sols[:referenced_samples],
            error_traces=error_traces,
        )
