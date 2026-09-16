"""Search directions strategy using SOLID architecture.

Architecture:
    Layer 1: RandomInputProvider or ArchiveInputProvider → generate solutions
    Layer 2: ComputeRhsTransform → compute RHS = A @ x
    Layer 3: PCG solver → run CG with search direction tracking
    Layer 4: Collect (A @ p_k, p_k) pairs from iteration history

This strategy collects (A @ p_k, p_k) pairs from CG iterations without requiring
exact solutions. Training mapping: NN(A @ p_k) ≈ p_k, meaning NN ≈ A^{-1}
"""

from __future__ import annotations

import numpy as np
from torchalg.models.result import SolverResult

from neuralls.domain.normalization import ResidualTraceSamples

from ..helpers import _build_trace_indices, resolve_trace_generation_counts
from ..interfaces import ArchiveData, ArchiveField, GeneratedSamples, TracingSolverCallable
from ..providers import HybridInputProvider
from ..runner import register_single_rhs_strategy
from ..strategy_configs import SearchDirectionsConfig
from ..trace_utils import _referenced_sample_count, _trim_residual_traces
from ..transforms import ComputeRhsTransform


def _direction_trace_to_numpy(info: SolverResult) -> np.ndarray:
    if info.direction_vectors is None:
        raise RuntimeError("Search directions were not captured by the torchalg solver result.")
    return info.direction_vectors.detach().cpu().numpy()


@register_single_rhs_strategy(supports_matrix_replacement=True)
class SearchDirectionsStrategy:
    """Collect search direction pairs from CG for neural preconditioner training.

    This strategy generates training data for learning A^{-1} without requiring
    exact solutions. It collects (A @ p_k, p_k) pairs from CG iterations where:
    - p_k: search direction at iteration k
    - A @ p_k: search direction product (computed in CG anyway)

    The neural network learns: NN(A @ p_k) ≈ p_k, i.e., NN ≈ A^{-1}

    This is useful for:
    - Training preconditioners for a single matrix A (run CG from different u_0)
    - Training preconditioners for a family of matrices (run on multiple systems)
    - Learning inverse operators without expensive exact solves
    """

    name = "search_directions"
    ConfigType = SearchDirectionsConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict,
        solver: TracingSolverCallable,
        single_rhs: np.ndarray | None = None,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate (A @ p_k, p_k) pairs from CG iterations.

        Supports two modes:
        - Single RHS mode: If single_rhs provided, run CG multiple times on the same RHS
        - Multiple RHS mode: If single_rhs is None, generate N different RHS vectors

        Args:
            matrix: System matrix
            cfg: Configuration dictionary
            single_rhs: Optional single RHS vector. If provided, all samples solve A @ x = single_rhs
            archive: Optional archive data to seed generation

        Returns:
            GeneratedSamples with search_directions_traces populated
        """
        # Validate and convert to typed config
        config = SearchDirectionsConfig(**cfg)

        window = config.window
        rng = np.random.default_rng(config.seed)
        available_systems: int | None = None

        if single_rhs is None and archive is not None:
            if archive.lhs is not None:
                available_systems = int(archive.lhs.shape[0])
            elif archive.rhs is not None:
                available_systems = int(archive.rhs.shape[0])

        num_base_systems, final_rows = resolve_trace_generation_counts(
            config.samples,
            window=window,
            available_systems=available_systems,
            strategy_name="search_directions",
        )

        n = matrix.shape[0]

        # Choose mode based on single_rhs parameter
        if single_rhs is not None:
            # Mode 1: Single RHS - run CG multiple times on the SAME RHS
            # Create array of identical RHS vectors for processing
            rhs_samples = np.tile(single_rhs, (num_base_systems, 1))
            # Note: We don't generate solutions in this mode (CG will solve from x0=0)
            sols = None
        else:
            # Mode 2: Multiple RHS - generate N different RHS vectors (SOLID pattern)
            # Layer 1: Input provision (archive with random fallback)
            solution_provider = HybridInputProvider(
                archive=archive, field=ArchiveField.LHS, scale=1.0
            )
            sols = solution_provider.provide(matrix, count=num_base_systems, rng=rng)

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

        direction_blocks: list[np.ndarray] = []
        product_blocks: list[np.ndarray] = []
        sample_indices: list[np.ndarray] = []
        iteration_indices: list[np.ndarray] = []

        for sample_idx, rhs_vec in enumerate(rhs_samples):
            _, info = solver(
                matrix,
                rhs_vec,
                np.zeros(n, dtype=np.float64),
                maxiter=window.stop,
                rtol=config.rtol,
                atol=config.atol,
            )

            direction_seq_full = _direction_trace_to_numpy(info)
            if direction_seq_full.size == 0:
                raise RuntimeError(f"Search directions array is empty for sample {sample_idx + 1}.")

            direction_seq, indices = window.select_with_indices(direction_seq_full)

            product_seq = np.array([matrix @ p for p in direction_seq], dtype=np.float64)

            direction_blocks.append(direction_seq)
            product_blocks.append(product_seq)
            sidx, iidx = _build_trace_indices(sample_idx, indices)
            sample_indices.append(sidx)
            iteration_indices.append(iidx)

        residual_traces = ResidualTraceSamples(
            residuals=np.vstack(product_blocks),
            solutions=np.vstack(direction_blocks),
            sample_indices=np.concatenate(sample_indices),
            iteration_indices=np.concatenate(iteration_indices),
            search_directions=np.vstack(direction_blocks),
            search_direction_products=np.vstack(product_blocks),
        )
        residual_traces = _trim_residual_traces(residual_traces, final_rows)
        referenced_samples = _referenced_sample_count(residual_traces.sample_indices)

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs_samples[:referenced_samples],
            solutions=(None if sols is None else sols[:referenced_samples]),
            residual_traces=residual_traces,
        )
