"""Eigenvector-based generation strategies.

This module implements forward and inverse eigenvector strategies for data
generation using the SOLID architecture pattern.

Architecture:
    - Forward mode: EigenvectorCombinationTransform → ComputeRhsTransform
      Generates solutions x (eigenvector combinations), then computes b = A @ x
    - Inverse mode: EigenvectorCombinationTransform → SolveTransform
      Generates RHS b (eigenvector combinations), then solves x = A^-1 @ b

Both strategies use the same transform for eigenvector generation,
differing only in the final transformation step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import ArchiveData, SolvedRows
from ..runner import register_strategy
from ..strategy_configs import EigenvectorForwardConfig, EigenvectorInverseConfig
from ..transforms import (
    ComputeRhsTransform,
    EigenvectorCombinationTransform,
    SolveTransform,
)


@register_strategy
class EigenvectorForwardStrategy:
    """Forward mode: generate solutions x, compute RHS b = A @ x.

    Architecture:
        Layer 1: EigenvectorCombinationTransform → generate solutions
        Layer 2: ComputeRhsTransform → compute RHS = A @ x

    This is the cheap direction - only requires matrix multiplication.
    Use when you have solutions (or can generate them) and need RHS vectors.

    Performance: O(n²) per sample (matrix-vector multiplication)
    """

    name = "eigenvector_forward"
    ConfigType = EigenvectorForwardConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> SolvedRows:
        """Generate samples: eigenvector combinations → compute RHS.

        Args:
            matrix: System matrix
            cfg: Configuration dictionary
            archive: Optional archive data (ignored)

        Returns:
            GeneratedSamples with eigenvector-based data
        """
        # Validate configuration
        config = EigenvectorForwardConfig(**cfg)

        # Layer 1: Generate eigenvector combinations (solutions)
        eigen_transform = EigenvectorCombinationTransform(
            count=config.samples,
            which=config.which,
            rng=np.random.default_rng(config.seed),
            num_eigenvectors=config.num_eigenvectors,
            include_eigenvectors=config.include_eigenvectors,
        )
        solutions = eigen_transform.transform(matrix)

        # Layer 2: Compute RHS = A @ x
        rhs_transform = ComputeRhsTransform(matrix)
        rhs = rhs_transform.transform(solutions)

        return SolvedRows(
            matrix=matrix,
            rhs=rhs,
            solutions=solutions,
        )


@register_strategy
class EigenvectorInverseStrategy:
    """Inverse mode: generate RHS b, solve for solutions x = A^-1 @ b.

    Architecture:
        Layer 1: EigenvectorCombinationTransform → generate RHS
        Layer 2: SolveTransform → solve x = A^-1 @ b

    This is the expensive direction - requires solving linear systems.
    Use when you have RHS vectors and need to compute solutions.

    Performance:
        - Direct solve: O(n³) per sample (Cholesky factorization)
        - CG solve: O(n² * iters) per sample (iterative)

    Configuration:
        Supports configurable solving via solve_config:
        - method: "direct" (default) or "cg"
        - tolerance: CG relative tolerance (default: 1e-12)
        - max_iters: CG max iterations (default: 500)
        - assume_pos_def: Assume positive-definite (default: True)
    """

    name = "eigenvector_inverse"
    ConfigType = EigenvectorInverseConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> SolvedRows:
        """Generate samples: eigenvector combinations → solve for solutions.

        Args:
            matrix: System matrix
            cfg: Configuration dictionary
            archive: Optional archive data (ignored)

        Returns:
            GeneratedSamples with eigenvector-based data
        """
        # Validate configuration
        config = EigenvectorInverseConfig(**cfg)

        # Layer 1: Generate eigenvector combinations (RHS)
        eigen_transform = EigenvectorCombinationTransform(
            count=config.samples,
            which=config.which,
            rng=np.random.default_rng(config.seed),
            num_eigenvectors=config.num_eigenvectors,
            include_eigenvectors=config.include_eigenvectors,
        )
        rhs = eigen_transform.transform(matrix)

        # Layer 2: Solve for solutions x = A^-1 @ b
        solved = SolveTransform(matrix, config.solve_config).transform(rhs)

        return SolvedRows(
            matrix=matrix,
            rhs=rhs,
            solutions=solved.solutions,
            notices=solved.unconverged,
        )
