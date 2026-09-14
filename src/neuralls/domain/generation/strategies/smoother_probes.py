"""Smoother-filtered probe strategy: random vectors damped by weighted-Jacobi sweeps.

Instead of deriving "algebraically smooth" error snapshots from actual CG
trajectories (see `residuals.py`), this strategy synthesizes them directly:
random probe vectors are passed through `steps` weighted-Jacobi damping
sweeps. A Jacobi smoother's error-propagation operator G = I - omega D^-1 A
quickly attenuates any direction it handles well, so what survives after
`steps` sweeps is, by construction, smoother-resistant — exactly the
directions a multigrid coarse-grid correction needs to cover.

This is the one strategy in this package that briefly leaves numpy for a
torch round-trip: the damping sweep itself (`apply_jacobi_damping`) lives in
`torchalg` so it exercises the same weighted-Jacobi map POD-2G's smoother
weighting (`composition/preconditioners/_weighting.py`) and the AMG smoother
actually run — reusing it here means "algebraically smooth" means the exact
same thing in both places, not two independently-tuned approximations of it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchalg.preconditioners.implementations.pod import apply_jacobi_damping

from ..interfaces import ArchiveData, GeneratedSamples
from ..runner import register_strategy
from ..strategy_configs import SmootherFilteredProbesConfig
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
    ``steps``/``omega`` rather than emerging incidentally from a CG
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
                precomputed input).

        Returns:
            GeneratedSamples: ``solutions`` are the damped probe vectors,
                ``rhs = matrix @ solutions``.
        """
        config = SmootherFilteredProbesConfig(**cfg)
        rng = np.random.default_rng(config.seed)

        probes = _sample_probes(rng, config.samples, matrix.shape[0], config.probe_distribution)
        probes = probes.astype(matrix.dtype, copy=False)

        matrix_t = torch.as_tensor(matrix)
        probes_t = torch.as_tensor(probes)
        damped = apply_jacobi_damping(probes_t, matrix_t, omega=config.omega, steps=config.steps)
        solutions = damped.numpy()

        rhs_transform = ComputeRhsTransform(matrix)
        rhs = rhs_transform.transform(solutions)

        return GeneratedSamples(matrix=matrix, rhs=rhs, solutions=solutions)
