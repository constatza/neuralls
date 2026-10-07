"""Interfaces and data containers for generation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from neuralls.domain.normalization import ErrorTraceSamples, ResidualTraceSamples
from neuralls.shared.types import SystemMatrix


class TracingSolverCallable(Protocol):
    """Callable that runs a tracing iterative solver.

    The returned second element must have .iteration_history populated.
    """

    def __call__(
        self,
        A: NDArray,
        b: NDArray,
        x0: NDArray,
        *,
        maxiter: int,
        rtol: float,
        atol: float,
    ) -> tuple[NDArray, Any]: ...


@dataclass(frozen=True)
class GeneratedSamples:
    """Container for generated data."""

    matrix: SystemMatrix
    rhs: np.ndarray | None
    solutions: np.ndarray | None
    residual_traces: ResidualTraceSamples | None = None
    error_traces: ErrorTraceSamples | None = None


class ArchiveField(StrEnum):
    """Closed set of ArchiveData field names, addressable via getattr-based dispatch."""

    LHS = "lhs"
    RHS = "rhs"


@dataclass(frozen=True)
class ArchiveData:
    """Container for pre-computed LHS solution and RHS vectors.

    Used when loading from solution archives instead of generating random data.

    Attributes:
        lhs: Solution vectors (x in A @ x = b), shape (N, n)
        rhs: Optional pre-computed RHS vectors (b), shape (N, n)
    """

    lhs: np.ndarray
    rhs: np.ndarray | None = None


@runtime_checkable
class MatrixGenerationStrategy(Protocol):
    """Protocol for strategies that generate from the matrix alone."""

    name: str
    ConfigType: object

    def generate(
        self,
        matrix: SystemMatrix,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate samples from matrix.

        Args:
            matrix: System matrix, shape (n, n)
            cfg: Strategy configuration dictionary (validated by strategy)
            archive: Optional pre-computed archive data

        Returns:
            GeneratedSamples with matrix, rhs, solutions, and optional traces
        """
        ...


@runtime_checkable
class SingleRhsGenerationStrategy(Protocol):
    """Protocol for strategies that support explicit single-RHS generation."""

    name: str
    ConfigType: object

    def generate(
        self,
        matrix: SystemMatrix,
        *,
        cfg: dict[str, Any],
        solver: TracingSolverCallable,
        single_rhs: np.ndarray | None = None,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate samples from matrix with an optional shared RHS."""
        ...
