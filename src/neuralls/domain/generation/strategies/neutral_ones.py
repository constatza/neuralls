"""Neutral ones strategy: x=ones, b=A@x for unbiased baseline testing using SOLID pattern.

Architecture:
    Layer 1 (Input): ConstantInputProvider generates x=ones solutions
    Layer 2 (Transform): ComputeRhsTransform computes b = A @ x
    Layer 3 (Strategy): Orchestrates provider and transform
"""

from __future__ import annotations

import numpy as np

from ..interfaces import ArchiveData, GeneratedSamples
from ..providers import ConstantInputProvider
from ..runner import register_strategy
from ..strategy_configs import NeutralOnesConfig
from ..transforms import ComputeRhsTransform


@register_strategy
class NeutralOnesStrategy:
    """Generate neutral test cases with x=ones, b=A@x.

    SOLID Pattern:
        - ConstantInputProvider: Generates x=ones solutions (Layer 1)
        - ComputeRhsTransform: Computes b = A @ x (Layer 2)
        - Strategy: Orchestrates provider and transform (Layer 3)

    This strategy creates a deterministic, unbiased test case where the solution
    is simply a vector of ones. This is useful for:
    - Consistent baseline comparison across different experiments
    - Testing solver behavior on simple, known solutions
    - Validating that A @ ones produces expected RHS

    The normalization is applied by the data generation pipeline BEFORE this
    strategy is called, so this generates normalized data in normalized space.
    """

    name = "neutral_ones"
    ConfigType = NeutralOnesConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate x=ones and compute b=A@x.

        Args:
            matrix: System matrix (already normalized)
            cfg: Configuration dictionary
            archive: Optional archive data (ignored)

        Returns:
            GeneratedSamples with x=ones solutions and b=A@x RHS
        """
        # Validate and convert to typed config
        config = NeutralOnesConfig(**cfg)

        rng = np.random.default_rng()  # Unused, for protocol compliance

        # Layer 1: Input provision (constant ones)
        provider = ConstantInputProvider(value=1.0)
        solutions = provider.provide(matrix, count=config.samples, rng=rng)

        # Layer 2: Transformation (compute RHS from solutions)
        transform = ComputeRhsTransform(matrix)
        rhs = transform.transform(solutions)

        return GeneratedSamples(matrix=matrix, rhs=rhs, solutions=solutions)
