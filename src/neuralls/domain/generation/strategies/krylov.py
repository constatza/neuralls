"""Krylov strategy implementation using SOLID architecture pattern.

Architecture:
    Layer 1: KrylovBasisTransform → generate solutions via Lanczos
    Layer 2: ComputeRhsTransform → compute RHS = A @ x

The Krylov strategy uses Lanczos iteration to build an orthonormal basis for
the Krylov subspace, then generates random linear combinations from this basis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import ArchiveData, SolvedRows
from ..runner import register_strategy
from ..strategy_configs import KrylovConfig
from ..transforms import ComputeRhsTransform, KrylovBasisTransform


@register_strategy
class KrylovStrategy:
    """Generate samples using Krylov subspace methods.

    Architecture:
        Layer 1: KrylovBasisTransform → generate solutions via Lanczos
        Layer 2: ComputeRhsTransform → compute RHS = A @ x

    The strategy builds a Krylov subspace K_m(A, v) using Lanczos iteration,
    then generates random linear combinations from the orthonormal basis.
    This produces solutions that span the Krylov subspace.

    Performance: O(m * n²) for Lanczos + O(count * n²) for RHS computation
    """

    name = "krylov"
    ConfigType = KrylovConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> SolvedRows:
        """Generate samples using Krylov subspace methods.

        Args:
            matrix: System matrix
            cfg: Configuration dictionary
            archive: Optional archive data (ignored)

        Returns:
            GeneratedSamples with Krylov-generated solutions
        """
        # Validate configuration
        config = KrylovConfig(**cfg)

        # Layer 1: Build Krylov basis and generate combinations (solutions)
        krylov_transform = KrylovBasisTransform(
            krylov_dim=config.krylov_iters,
            num_samples=config.samples,
            rng=np.random.default_rng(config.seed),
        )
        solutions = krylov_transform.transform(matrix)

        # Layer 2: Compute RHS = A @ x
        rhs_transform = ComputeRhsTransform(matrix)
        rhs = rhs_transform.transform(solutions)

        return SolvedRows(matrix=matrix, rhs=rhs, solutions=solutions)
