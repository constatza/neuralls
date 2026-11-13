"""Interfaces and data containers for generation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from neuralls.domain.normalization import ErrorTraceSamples, ResidualTraceSamples
from neuralls.shared.types import SystemMatrix

from .notices import GenerationNotice
from .ports import ArchiveSource


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


@dataclass(frozen=True, slots=True)
class SolvedRows:
    """(RHS, solution) pairs, one training row each.

    Attributes:
        matrix: System matrix the rows belong to.
        rhs: Feature rows, shape (rows, n).
        solutions: Target rows, shape (rows, n).
        notices: Outcomes of producing the rows (solve non-convergence, kept invalid pairs).
    """

    matrix: SystemMatrix
    rhs: np.ndarray
    solutions: np.ndarray
    notices: tuple[GenerationNotice, ...] = ()

    @property
    def row_count(self) -> int:
        """Training rows emitted."""
        return int(self.rhs.shape[0])


@dataclass(frozen=True, slots=True)
class ResidualTraceRows:
    """Smoother-probe traces: the training rows are (A @ p_k, p_k).

    Attributes:
        matrix: System matrix the traces belong to.
        residual_traces: The traced residuals and probe vectors.
    """

    matrix: SystemMatrix
    residual_traces: ResidualTraceSamples

    @property
    def row_count(self) -> int:
        """Training rows emitted."""
        return int(self.residual_traces.residuals.shape[0])


@dataclass(frozen=True, slots=True)
class ErrorTraceRows:
    """CG trajectory traces: the training rows are (r_k, e_k).

    Attributes:
        matrix: System matrix the traces belong to.
        rhs: Base-system right-hand sides the traces start from, shape (systems, n).
        solutions: Base-system true solutions, shape (systems, n).
        error_traces: The traced residuals and errors.
    """

    matrix: SystemMatrix
    rhs: np.ndarray
    solutions: np.ndarray
    error_traces: ErrorTraceSamples

    @property
    def row_count(self) -> int:
        """Training rows emitted."""
        return int(self.error_traces.errors.shape[0])


type GeneratedSamples = SolvedRows | ResidualTraceRows | ErrorTraceRows
"""What one strategy run emits; each variant is one training-row shape."""


class ArchiveField(StrEnum):
    """Closed set of ArchiveData field names, addressable via getattr-based dispatch."""

    LHS = "lhs"
    RHS = "rhs"


@dataclass(frozen=True, slots=True)
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
class ArchiveGenerationStrategy(Protocol):
    """Protocol for strategies that draw their vectors from archive files."""

    name: str
    ConfigType: object

    def generate(
        self,
        matrix: SystemMatrix,
        *,
        cfg: dict[str, Any],
        archive_source: ArchiveSource,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate samples from matrix and the archive files ``cfg`` names."""
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
        archive_source: ArchiveSource,
        single_rhs: np.ndarray | None = None,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate samples from matrix with an optional shared RHS."""
        ...
