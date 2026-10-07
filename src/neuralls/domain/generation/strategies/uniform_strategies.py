"""Uniform distribution generation strategies.

Architecture:
    Forward mode: UniformInputProvider → ComputeRhsTransform
        Generates solutions x ~ U(a, b), then computes RHS b = A @ x.
    Inverse mode: UniformInputProvider → SolveTransform
        Generates RHS b ~ U(a, b), then solves x = A^-1 @ b.

Both strategies use UniformInputProvider, differing only in the final transform.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import ArchiveData, GeneratedSamples
from ..providers import UniformInputProvider
from ..runner import register_strategy
from ..strategy_configs import UniformForwardConfig, UniformInverseConfig
from ..transforms import ComputeRhsTransform, SolveTransform


@register_strategy
class UniformForwardStrategy:
    """Forward mode: generate solutions x ~ U(a, b), compute RHS b = A @ x.

    Architecture:
        Layer 1: UniformInputProvider → generate solutions
        Layer 2: ComputeRhsTransform → compute RHS = A @ x

    Use when you want solutions uniformly distributed in a bounded interval.
    """

    name = "uniform_forward"
    ConfigType = UniformForwardConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate uniform solutions and compute RHS.

        Args:
            matrix: System matrix (already normalized).
            cfg: Configuration dictionary.
            archive: Optional archive data (ignored).

        Returns:
            GeneratedSamples with uniform solutions and b = A @ x RHS.

        Raises:
            ValueError: If a >= b in the configuration.
        """
        config = UniformForwardConfig(**cfg)
        if config.a >= config.b:
            raise ValueError(f"'a' must be less than 'b', got a={config.a}, b={config.b}.")
        rng = np.random.default_rng(config.seed)

        # Layer 1: Input provision (Uniform solutions)
        provider = UniformInputProvider(a=config.a, b=config.b)
        solutions = provider.provide(matrix, count=config.samples, rng=rng)

        # Layer 2: Transformation (compute RHS from solutions)
        rhs = ComputeRhsTransform(matrix).transform(solutions)

        return GeneratedSamples(matrix=matrix, rhs=rhs, solutions=solutions)


@register_strategy
class UniformInverseStrategy:
    """Inverse mode: generate RHS b ~ U(a, b), solve for x = A^-1 @ b.

    Architecture:
        Layer 1: UniformInputProvider → generate RHS vectors
        Layer 2: SolveTransform → solve x = A^-1 @ b

    Use when you need uniformly distributed RHS vectors with exact solutions.
    """

    name = "uniform_inverse"
    ConfigType = UniformInverseConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate uniform RHS vectors and solve for solutions.

        Args:
            matrix: System matrix (already normalized).
            cfg: Configuration dictionary.
            archive: Optional archive data (ignored).

        Returns:
            GeneratedSamples with uniform RHS vectors and solved solutions.

        Raises:
            ValueError: If a >= b in the configuration.
        """
        config = UniformInverseConfig(**cfg)
        if config.a >= config.b:
            raise ValueError(f"'a' must be less than 'b', got a={config.a}, b={config.b}.")
        rng = np.random.default_rng(config.seed)

        # Layer 1: Input provision (Uniform RHS)
        provider = UniformInputProvider(a=config.a, b=config.b)
        rhs = provider.provide(matrix, count=config.samples, rng=rng)

        # Layer 2: Transformation (solve for solutions)
        solutions = SolveTransform(matrix, **config.solve_config.model_dump()).transform(rhs)

        return GeneratedSamples(matrix=matrix, rhs=rhs, solutions=solutions)
