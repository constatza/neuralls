"""Constant vector generation strategies.

Architecture:
    Forward mode: ConstantInputProvider → ComputeRhsTransform
        Generates solutions x = ones * c, then computes RHS b = A @ x.
    Inverse mode: ConstantInputProvider → SolveTransform
        Generates RHS b = ones * c, then solves x = A^-1 @ b.

Both strategies use ConstantInputProvider, differing only in the final transform.
This provides a fully deterministic, parameterised alternative to neutral_ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import ArchiveData, SolvedRows
from ..providers import ConstantInputProvider
from ..runner import register_strategy
from ..strategy_configs import ConstantForwardConfig, ConstantInverseConfig
from ..transforms import ComputeRhsTransform, SolveTransform


@register_strategy
class ConstantForwardStrategy:
    """Forward mode: generate solutions x = ones * c, compute RHS b = A @ x.

    Architecture:
        Layer 1: ConstantInputProvider → generate constant solution vectors
        Layer 2: ComputeRhsTransform → compute RHS = A @ x

    Extends neutral_ones with a configurable constant c.
    """

    name = "constant_forward"
    ConfigType = ConstantForwardConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> SolvedRows:
        """Generate constant solutions and compute RHS.

        Args:
            matrix: System matrix (already normalized).
            cfg: Configuration dictionary.
            archive: Optional archive data (ignored).

        Returns:
            GeneratedSamples with constant solutions x = ones * c and b = A @ x.
        """
        config = ConstantForwardConfig(**cfg)
        rng = np.random.default_rng(config.seed)

        # Layer 1: Input provision (constant solutions)
        provider = ConstantInputProvider(value=config.c)
        solutions = provider.provide(matrix, count=config.samples, rng=rng)

        # Layer 2: Transformation (compute RHS from solutions)
        rhs = ComputeRhsTransform(matrix).transform(solutions)

        return SolvedRows(matrix=matrix, rhs=rhs, solutions=solutions)


@register_strategy
class ConstantInverseStrategy:
    """Inverse mode: generate RHS b = ones * c, solve for x = A^-1 @ b.

    Architecture:
        Layer 1: ConstantInputProvider → generate constant RHS vectors
        Layer 2: SolveTransform → solve x = A^-1 @ b

    Produces the exact solution to A @ x = c * ones.
    """

    name = "constant_inverse"
    ConfigType = ConstantInverseConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> SolvedRows:
        """Generate constant RHS vectors and solve for solutions.

        Args:
            matrix: System matrix (already normalized).
            cfg: Configuration dictionary.
            archive: Optional archive data (ignored).

        Returns:
            GeneratedSamples with constant RHS b = ones * c and solved solutions.
        """
        config = ConstantInverseConfig(**cfg)
        rng = np.random.default_rng(config.seed)

        # Layer 1: Input provision (constant RHS)
        provider = ConstantInputProvider(value=config.c)
        rhs = provider.provide(matrix, count=config.samples, rng=rng)

        # Layer 2: Transformation (solve for solutions)
        solved = SolveTransform(matrix, config.solve_config).transform(rhs)

        return SolvedRows(
            matrix=matrix, rhs=rhs, solutions=solved.solutions, notices=solved.unconverged
        )
