"""Smoother-filtered probe strategy: random vectors damped by weighted-Jacobi sweeps.

Instead of deriving "algebraically smooth" error snapshots from actual CG
trajectories (see `residuals.py`), this strategy synthesizes them directly:
random probe vectors are passed through `window.stop` weighted-Jacobi damping
sweeps, and `window` selects which of those sweeps are kept (by default just
the final, fully-damped one). A Jacobi smoother's error-propagation operator
G = I - omega D^-1 A quickly attenuates any direction it handles well, so
what survives after `window.stop` sweeps is, by construction, smoother-
resistant — exactly the directions a multigrid coarse-grid correction needs
to cover.

This is the one strategy in this package that briefly leaves numpy for a
torch round-trip: the damping sweep itself (`apply_jacobi_damping_trajectory`)
lives in `torchalg` so it exercises the same weighted-Jacobi map POD-2G's
smoother weighting (`composition/preconditioners/_weighting.py`) and the AMG
smoother actually run — reusing it here means "algebraically smooth" means
the exact same thing in both places, not two independently-tuned
approximations of it.

Output is a `ResidualTraceSamples` (the same trace container
`search_directions.py` populates) — always 2D (`sample_indices`/
`iteration_indices` paired rows), whether `window` keeps one sweep per probe
(the default) or several, so downstream consumers see a consistent shape
regardless of how many steps were kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchalg.preconditioners.implementations.pod import apply_jacobi_damping_trajectory

from neuralls.domain.normalization import ResidualTraceSamples

from ..helpers import _build_trace_indices, resolve_trace_generation_counts
from ..interfaces import ArchiveData, GeneratedSamples
from ..runner import register_strategy
from ..strategy_configs import SmootherFilteredProbesConfig
from ..trace_utils import _trim_residual_traces
from ..transforms import ComputeRhsTransform


def _sample_probes(
    rng: np.random.Generator, count: int, dimension: int, distribution: str
) -> np.ndarray:
    """Draw ``count`` random probe vectors of ``dimension``, undamped.

    Args:
        rng: Seeded random generator.
        count: Number of probe vectors to draw.
        dimension: Vector length (matches the system matrix size).
        distribution: ``"gaussian"`` (standard normal) or ``"rademacher"``
            (uniform +-1 entries).

    Returns:
        np.ndarray: Shape (count, dimension), dtype float64.
    """
    if distribution == "rademacher":
        return rng.choice([-1.0, 1.0], size=(count, dimension))
    return rng.standard_normal((count, dimension))


@register_strategy
class SmootherFilteredProbesStrategy:
    """Generate error snapshots by Jacobi-damping random probe vectors.

    Produces snapshots whose "smoothness" is controlled directly by
    ``window``/``omega`` rather than emerging incidentally from a CG
    trajectory — useful as a POD-2G training ensemble targeted specifically
    at the modes a weighted-Jacobi smoother struggles with.
    """

    name = "smoother_filtered_probes"
    ConfigType = SmootherFilteredProbesConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate smoother-filtered probe snapshots and their RHS.

        Args:
            matrix: System matrix (already normalized).
            cfg: Configuration dictionary, validated against
                ``SmootherFilteredProbesConfig``.
            archive: Optional archive data (ignored — this strategy needs no
                precomputed input, and has no finite archive-backed source,
                so ``samples=-1`` is not supported).

        Returns:
            GeneratedSamples: ``residual_traces`` holds the damped probe
                vectors (``solutions``) and their RHS (``residuals``,
                ``rhs = matrix @ solutions``), one row per kept sweep per
                probe.
        """
        config = SmootherFilteredProbesConfig(**cfg)
        window = config.window
        rng = np.random.default_rng(config.seed)

        num_base_probes, final_rows = resolve_trace_generation_counts(
            config.samples,
            window=window,
            available_systems=None,
            strategy_name=self.name,
        )

        probes = _sample_probes(rng, num_base_probes, matrix.shape[0], config.probe_distribution)
        probes = probes.astype(matrix.dtype, copy=False)

        matrix_t = torch.as_tensor(matrix)
        probes_t = torch.as_tensor(probes)
        trajectories = apply_jacobi_damping_trajectory(
            probes_t, matrix_t, omega=config.omega, steps=window.stop
        )

        rhs_transform = ComputeRhsTransform(matrix)

        solution_blocks: list[np.ndarray] = []
        rhs_blocks: list[np.ndarray] = []
        sample_indices: list[np.ndarray] = []
        iteration_indices: list[np.ndarray] = []

        for sample_idx, trajectory in enumerate(trajectories):
            selected, indices = window.select_with_indices(trajectory.numpy())
            solution_blocks.append(selected)
            rhs_blocks.append(rhs_transform.transform(selected))
            sidx, iidx = _build_trace_indices(sample_idx, indices)
            sample_indices.append(sidx)
            iteration_indices.append(iidx)

        residual_traces = ResidualTraceSamples(
            residuals=np.vstack(rhs_blocks),
            solutions=np.vstack(solution_blocks),
            sample_indices=np.concatenate(sample_indices),
            iteration_indices=np.concatenate(iteration_indices),
        )
        residual_traces = _trim_residual_traces(residual_traces, final_rows)

        return GeneratedSamples(
            matrix=matrix,
            rhs=None,
            solutions=None,
            residual_traces=residual_traces,
        )
