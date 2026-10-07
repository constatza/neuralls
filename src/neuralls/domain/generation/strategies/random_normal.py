"""Random/normal strategy implementation using SOLID provider + transform pattern.

Architecture:
    Layer 1 (Input): RandomInputProvider generates random normal solutions
    Layer 2 (Transform): ComputeRhsTransform computes b = A @ x
    Layer 3 (Strategy): Orchestrates provider and transform
"""

from __future__ import annotations

import numpy as np

from ..interfaces import ArchiveData, GeneratedSamples
from ..providers import RandomInputProvider
from ..runner import register_strategy
from ..strategy_configs import RandomNormalConfig
from ..transforms import ComputeRhsTransform


@register_strategy
class RandomNormalStrategy:
    """Generate random normal solutions and compute RHS.

    SOLID Pattern:
        - RandomInputProvider: Generates random solutions (Layer 1)
        - ComputeRhsTransform: Computes b = A @ x (Layer 2)
        - Strategy: Orchestrates provider and transform (Layer 3)

    This decoupling enables:
        - Replacing random input with archive/file input
        - Reusing ComputeRhsTransform across strategies
        - Testing provider and transform independently
    """

    name = "random"
    ConfigType = RandomNormalConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate random normal solutions and compute RHS.

        Args:
            matrix: System matrix
            cfg: Configuration dictionary
            archive: Optional archive data (ignored)

        Returns:
            GeneratedSamples with random normal solutions
        """
        # Validate and convert to typed config
        config = RandomNormalConfig(**cfg)

        # Layer 1: Input provision (random generation)
        provider = RandomInputProvider(seed=config.seed, scale=config.target_rhs_scale)
        rng = np.random.default_rng(config.seed)
        solutions = provider.provide(matrix, count=config.samples, rng=rng)

        # Layer 2: Transformation (compute RHS from solutions)
        transform = ComputeRhsTransform(matrix)
        rhs = transform.transform(solutions)

        return GeneratedSamples(matrix=matrix, rhs=rhs, solutions=solutions)


# Alias for "normal"
@register_strategy
class NormalStrategy(RandomNormalStrategy):
    name = "normal"
