"""Residual trace strategies with error targets.

Architecture:
    Layer 1: Archive/File or Gaussian providers → generate true solutions
    Layer 2: ComputeRhsTransform → compute RHS = A @ x
    Layer 3: SciPyCGSolver → run CG with iteration history tracking
    Layer 4: Collect residual and error traces from iteration history

These strategies run CG and collect residual vectors together with exact error
targets dx_k = x_true - x_k at each iteration.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from loguru import logger

from neuralls.domain.normalization import ErrorTraceSamples

from ..helpers import _build_trace_indices, resolve_trace_generation_counts
from ..interfaces import ArchiveData, ArchiveField, GeneratedSamples, TracingSolverCallable
from ..providers import HybridInputProvider, RandomInputProvider, provide_solutions
from ..runner import register_single_rhs_strategy
from ..strategy_configs import ResidualErrorConfig
from ..trace_utils import _referenced_sample_count, _trim_error_traces
from ..transforms import ComputeRhsTransform


def _tensor_trace_to_numpy(value: torch.Tensor | None) -> np.ndarray:
    if value is None:
        return np.array([])
    return value.detach().cpu().numpy()


class _BaseResidualsStrategy:
    ConfigType = ResidualErrorConfig
    name: str

    def _resolve_available_systems(
        self,
        matrix: np.ndarray,
        config: ResidualErrorConfig,
        archive: ArchiveData | None,
        rng: np.random.Generator,
    ) -> int | None:
        if config.solutions_glob is not None:
            return len(
                provide_solutions(
                    matrix,
                    -1,
                    rng,
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
        archive: ArchiveData | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return provide_solutions(
            matrix,
            count,
            rng,
            solutions_glob=config.solutions_glob,
            archive=archive,
            shuffle=config.shuffle,
            seed=config.seed,
            strategy_name=self.name,
        )

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict,
        solver: TracingSolverCallable,
        single_rhs: np.ndarray | None = None,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate samples with full residual error traces.

        Supports two modes:
        - Single RHS mode: If single_rhs provided, run CG multiple times on the same RHS
        - Multiple RHS mode: If single_rhs is None, generate N different RHS vectors

        Args:
            matrix: System matrix
            cfg: Configuration dictionary
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
            available_systems = self._resolve_available_systems(matrix, config, archive, rng)

        num_base_systems, final_rows = resolve_trace_generation_counts(
            config.samples,
            window=window,
            available_systems=available_systems,
            strategy_name=self.name,
        )

        n = matrix.shape[0]

        # Choose mode based on single_rhs parameter
        if single_rhs is not None:
            # Mode 1: Single RHS - run CG multiple times on the SAME RHS
            # Solve exactly to get "true" solution for error target computation
            true_sol = np.linalg.solve(matrix, single_rhs)

            # Create array of identical RHS and solution vectors
            rhs_samples = np.tile(single_rhs, (num_base_systems, 1))
            sols = np.tile(true_sol, (num_base_systems, 1))
        else:
            sols = self._provide_true_solutions(
                matrix,
                num_base_systems,
                config,
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

        residual_blocks: list[np.ndarray] = []
        solution_current_blocks: list[np.ndarray] = []
        error_blocks: list[np.ndarray] = []
        sample_indices: list[np.ndarray] = []
        iteration_indices: list[np.ndarray] = []

        log_every = max(1, num_base_systems // 10)
        loop_start = time.monotonic()
        for sample_idx, (rhs_vec, true_sol) in enumerate(zip(rhs_samples, sols)):
            if sample_idx > 0 and sample_idx % log_every == 0:
                elapsed = time.monotonic() - loop_start
                logger.info(
                    f"{self.name}: {sample_idx}/{num_base_systems} samples "
                    f"({sample_idx / elapsed:.1f} samples/s, {elapsed:.1f}s elapsed)"
                )
            _, info = solver(
                matrix,
                rhs_vec,
                np.zeros(n, dtype=np.float64),
                maxiter=window.stop,
                rtol=config.rtol,
                atol=config.atol,
            )

            residual_seq_full = _tensor_trace_to_numpy(info.residual_vectors)
            solution_seq_full = _tensor_trace_to_numpy(info.solution_vectors)

            residual_seq, indices = window.select_with_indices(residual_seq_full)
            solution_seq = (
                window.select(solution_seq_full)
                if solution_seq_full.size > 0
                else solution_seq_full
            )

            error_seq = np.array([true_sol - x_k for x_k in solution_seq], dtype=np.float64)

            residual_blocks.append(residual_seq)
            solution_current_blocks.append(solution_seq)
            error_blocks.append(error_seq)
            sidx, iidx = _build_trace_indices(sample_idx, indices)
            sample_indices.append(sidx)
            iteration_indices.append(iidx)

        error_traces = ErrorTraceSamples(
            residuals=np.vstack(residual_blocks),
            solutions_current=np.vstack(solution_current_blocks),
            errors=np.vstack(error_blocks),
            true_solutions=sols,
            sample_indices=np.concatenate(sample_indices),
            iteration_indices=np.concatenate(iteration_indices),
        )
        error_traces = _trim_error_traces(error_traces, final_rows)
        referenced_samples = _referenced_sample_count(error_traces.sample_indices)

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs_samples[:referenced_samples],
            solutions=sols[:referenced_samples],
            error_traces=error_traces,
        )


@register_single_rhs_strategy(supports_matrix_replacement=True)
class ResidualsStrategy(_BaseResidualsStrategy):
    name = "residuals"


@register_single_rhs_strategy(supports_matrix_replacement=True)
class GaussianResidualsStrategy(_BaseResidualsStrategy):
    name = "gaussian_residuals"

    def _provide_true_solutions(
        self,
        matrix: np.ndarray,
        count: int,
        config: ResidualErrorConfig,
        archive: ArchiveData | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        provider = RandomInputProvider(seed=config.seed, scale=1.0)
        return provider.provide(matrix, count=count, rng=rng)
