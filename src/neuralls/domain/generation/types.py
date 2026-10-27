"""Data structures for generation strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuralls.domain.normalization import ErrorTraceSamples, ResidualTraceSamples

from .interfaces import ArchiveData, GeneratedSamples

# =============================================================================
# CONSTANTS
# =============================================================================

# Eigenvector strategy constants
EIGENVECTOR_SOLUTION_TOLERANCE = 1e-14  # Near machine precision for float64


# =============================================================================
# DATA LAYER: Immutable data structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class StrategyOutput:
    """Container for samples produced by a generation strategy.

    Attributes:
        rhs: Right-hand side vectors, shape (N, n)
        solutions: Solution vectors, shape (N, n)
        residual_traces: Optional residual trace samples from CG iterations
        error_traces: Optional error trace samples from CG iterations
    """

    rhs: np.ndarray
    solutions: np.ndarray
    residual_traces: ResidualTraceSamples | None = None
    error_traces: ErrorTraceSamples | None = None


@dataclass(frozen=True, slots=True)
class RawSamples:
    """Container for dataset samples (matrix + RHS/solution pairs).

    High-level data structure for generated datasets.
    Strategy-specific details (residual traces, mother RHS, etc.) are
    implementation details and should not leak into this interface.

    Attributes:
        matrix: System matrix, shape (n, n)
        rhs: RHS vectors (features), shape (N, n)
        solutions: Solution vectors (targets), shape (N, n)
    """

    matrix: np.ndarray
    rhs: np.ndarray
    solutions: np.ndarray


__all__ = [
    "EIGENVECTOR_SOLUTION_TOLERANCE",
    "ArchiveData",
    "GeneratedSamples",
    "RawSamples",
    "StrategyOutput",
]
