"""Gaussian distribution generation strategies.

Architecture:
    Forward mode: GaussianInputProvider → ComputeRhsTransform
        Generates solutions x ~ N(mu, sigma), then computes b = A @ x.
    Inverse mode: GaussianInputProvider → SolveTransform
        Generates RHS b ~ N(mu, sigma), then solves x = A^-1 @ b.

Both strategies use GaussianInputProvider, differing only in the final transform.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import ArchiveData, GeneratedSamples
from ..providers import GaussianInputProvider
from ..runner import register_strategy
from ..strategy_configs import GaussianForwardConfig, GaussianInverseConfig
from ..transforms import ComputeRhsTransform, SolveTransform


@register_strategy
class GaussianForwardStrategy:
    """Forward mode: generate solutions x ~ N(mu, sigma), compute RHS b = A @ x.

    Architecture:
        Layer 1: GaussianInputProvider → generate solutions
        Layer 2: ComputeRhsTransform → compute RHS = A @ x

    Use when you need neutral, unbiased training data with a known solution distribution.
    """

    name = "gaussian_forward"
    ConfigType = GaussianForwardConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate Gaussian solutions and compute RHS.

        Args:
            matrix: System matrix (already normalized).
            cfg: Configuration dictionary.
            archive: Optional archive data (ignored).

        Returns:
            GeneratedSamples with Gaussian solutions and b = A @ x RHS.
        """
        config = GaussianForwardConfig(**cfg)
        rng = np.random.default_rng(config.seed)

        # Layer 1: Input provision (Gaussian solutions)
        provider = GaussianInputProvider(mu=config.mu, sigma=config.sigma)
        solutions = provider.provide(matrix, count=config.samples, rng=rng)

        # Layer 2: Transformation (compute RHS from solutions)
        rhs = ComputeRhsTransform(matrix).transform(solutions)

        return GeneratedSamples(matrix=matrix, rhs=rhs, solutions=solutions)


@register_strategy
class GaussianInverseStrategy:
    """Inverse mode: generate RHS b ~ N(mu, sigma), solve for x = A^-1 @ b.

    Architecture:
        Layer 1: GaussianInputProvider → generate RHS vectors
        Layer 2: SolveTransform → solve x = A^-1 @ b

    Use when you need neutral RHS vectors with exact solutions for training.
    """

    name = "gaussian_inverse"
    ConfigType = GaussianInverseConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate Gaussian RHS vectors and solve for solutions.

        Args:
            matrix: System matrix (already normalized).
            cfg: Configuration dictionary.
            archive: Optional archive data (ignored).

        Returns:
            GeneratedSamples with Gaussian RHS vectors and solved solutions.
        """
        config = GaussianInverseConfig(**cfg)
        rng = np.random.default_rng(config.seed)

        # Layer 1: Input provision (Gaussian RHS)
        provider = GaussianInputProvider(mu=config.mu, sigma=config.sigma)
        rhs = provider.provide(matrix, count=config.samples, rng=rng)

        # Layer 2: Transformation (solve for solutions)
        solutions = SolveTransform(matrix, **config.solve_config.model_dump()).transform(rhs)

        return GeneratedSamples(matrix=matrix, rhs=rhs, solutions=solutions)
