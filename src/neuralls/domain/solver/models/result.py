"""Result dataclasses for solver execution.

This module defines result containers returned by solvers after execution.
Includes diagnostic information, convergence status, and optional traces.

Design:
    - SolverResult: General solver result with SciPy-aligned metadata
    - CGComparisonResult: Extended result for comparison experiments
    - IterationContext: Context passed to flexible preconditioners

Theory:
    Results contain both numerical outcomes (solution, residual) and diagnostic
    information (iterations, convergence status, breakdown flags). This separation
    enables post-analysis and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from neuralls.shared.types import ComparisonRhsSourceKind

    from .config import ComparisonGeneral


@dataclass
class SolverResult:
    """Execution result for iterative solvers with diagnostic metadata.

    This dataclass contains all information about a solver run:
    - Convergence status and final residual
    - Iteration counts and history
    - Diagnostic flags (breakdown, divergence)
    - Optional traces (residual vectors, solution vectors)

    Attributes:
        converged: Whether convergence criterion satisfied.
        iterations: Number of iterations executed.
        residual: Relative residual norm ||r||/||b|| at termination.
        residual_abs: Absolute residual norm ||r|| at termination.
        rhs_norm: Right-hand side norm ||b||.
        breakdown: Whether numerical breakdown occurred.
        info: Status code (0=success, >0=warning, <0=error).
        residual_history_rel: Relative residual norms across iterations.
        residual_history_abs: Absolute residual norms across iterations.
        tol: Tolerance used (for diagnostics).
        atol: Absolute tolerance used.
        maxiter: Maximum iterations allowed.
        event_log: Optional EventLog with discrete solver events.
        iteration_history: Optional IterationHistory with continuous iteration diagnostics.
        stopping_criterion: Description of why solver stopped.
        residual_vectors: Optional full residual vectors (shape: iterations x n).
        solution_vectors: Optional full solution vectors (shape: iterations x n).

    Example:
        >>> result = SolverResult(
        ...     converged=True,
        ...     iterations=10,
        ...     residual=1e-8,
        ...     residual_abs=1e-8,
        ...     rhs_norm=1.0,
        ...     breakdown=False,
        ... )
        >>> result.converged
        True
    """

    converged: bool
    """Whether convergence criterion satisfied: ||r|| <= max(rtol*||b||, atol)."""

    iterations: int
    """Number of iterations executed."""

    residual: float
    """Relative residual norm ||r||/||b|| at termination."""

    residual_abs: float
    """Absolute residual norm ||r|| at termination."""

    rhs_norm: float
    """Right-hand side norm ||b||."""

    breakdown: bool
    """Whether numerical breakdown occurred (NaN/Inf). Terminal condition."""

    info: int = 0
    """Status code: 0=success, >0=warning (max iter), <0=error (breakdown)."""

    residual_history_rel: list[float] | None = None
    """Relative residual norms ||r_k||/||b|| across iterations."""

    residual_history_abs: list[float] | None = None
    """Absolute residual norms ||r_k|| across iterations."""

    tol: float | None = None
    """Relative tolerance used (for diagnostics)."""

    atol: float | None = None
    """Absolute tolerance used (for diagnostics)."""

    maxiter: int | None = None
    """Maximum iterations allowed."""

    event_log: Any | None = None
    """Optional algorithm diagnostics retained for legacy export/reporting payloads."""

    iteration_history: Any | None = None
    """Optional iteration telemetry retained for legacy export/reporting payloads."""

    stopping_criterion: str | None = None
    """Description of why solver stopped (e.g., 'converged', 'max_iter', 'breakdown')."""

    residual_vectors: np.ndarray | None = None
    """Optional full residual vectors (shape: iterations x n). Heavy, use sparingly."""

    solution_vectors: np.ndarray | None = None
    """Optional full solution vectors (shape: iterations x n). Heavy, use sparingly."""


@dataclass(slots=True)
class CGComparisonResult:
    """Result container for CG comparison experiments.

    Extended result class for comparison workflows that test multiple
    warm start and preconditioner combinations.

    Attributes:
        x: Final solution vector.
        converged: Whether convergence criterion satisfied.
        iterations: Number of iterations executed.
        residual: Relative residual norm at termination.
        residual_abs: Absolute residual norm at termination.
        residual_history_rel: Relative residual norms across iterations.
        residual_history_abs: Absolute residual norms across iterations.
        warm_start: Warm start strategy name.
        preconditioner: Preconditioner strategy name.
        step_helper: Step helper strategy name (e.g., 'none', 'neural').
        initial_guess: Initial guess vector x_0.
        exact_error: Error ||x - x_exact|| if exact solution known.
        rhs_norm: Right-hand side norm ||b||.
        breakdown: Whether numerical breakdown occurred.
        error_history_a_rel: Relative energy-norm errors across iterations.
        error_bound_a_rel: Golub-Meurant lower bound on the relative energy-norm
            error, used when the exact value is unavailable.
        helper_iterations: Iterations where step helper was invoked.
        helper_norms: Residual norms after step helper application.
        error: Error message if solver failed.

    Example:
        >>> result = CGComparisonResult(
        ...     x=np.zeros(10),
        ...     converged=True,
        ...     iterations=5,
        ...     residual=1e-8,
        ...     residual_abs=1e-8,
        ...     residual_history_rel=[1.0, 0.1, 0.01, 0.001, 1e-8],
        ...     residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        ...     preconditioner="jacobi",
        ...     initial_guess=np.zeros(10),
        ...     exact_error=None,
        ...     rhs_norm=1.0,
        ...     breakdown=False,
        ... )
        >>> result.preconditioner
        'jacobi'
    """

    x: np.ndarray
    """Final solution vector."""

    converged: bool
    """Whether convergence criterion satisfied."""

    iterations: int
    """Number of iterations executed."""

    residual: float
    """Relative residual norm at termination."""

    residual_abs: float
    """Absolute residual norm at termination."""

    residual_history_rel: list[float]
    """Relative residual norms ||r_k||/||b|| across iterations."""

    residual_history_abs: list[float]
    """Absolute residual norms ||r_k|| across iterations."""

    preconditioner: str
    """Preconditioner strategy name (e.g., 'none', 'jacobi', 'neural')."""

    initial_guess: np.ndarray
    """Initial guess vector x_0."""

    exact_error: float | None
    """Error ||x - x_exact|| if exact solution known."""

    rhs_norm: float
    """Right-hand side norm ||b||."""

    breakdown: bool
    """Whether numerical breakdown occurred."""

    error: str | None = None
    """Error message if solver failed."""

    error_history_a_rel: list[float] | None = None
    """Energy-norm errors ||e_k||_A / ||e_0||_A across iterations (None if unavailable)."""

    error_bound_a_rel: list[float] | None = None
    """Golub-Meurant lower bound on ||e_k||_A / ||e_0||_A, used when the exact
    energy error is unavailable (no reference solution). None if neither is available."""


@dataclass(frozen=True, slots=True)
class IterationContext:
    """Context passed to flexible preconditioners and step helpers.

    Provides iteration-specific information to preconditioners and helpers
    that need access to current solver state beyond just the residual.

    Attributes:
        iteration: Current iteration number (0-indexed).
        residual: Current residual vector r_k.
        solution: Current solution vector x_k.
        matrix: System matrix A (optional, may be expensive to store).
        rhs: Right-hand side vector b.

    Example:
        >>> import numpy as np
        >>> ctx = IterationContext(
        ...     iteration=5,
        ...     residual=np.ones(10),
        ...     solution=np.zeros(10),
        ...     matrix=np.eye(10),
        ...     rhs=np.ones(10),
        ... )
        >>> ctx.iteration
        5

    Usage:
        Context-aware preconditioners can use iteration info for adaptive behavior:

        >>> def adaptive_precond(r: np.ndarray, ctx: IterationContext) -> np.ndarray:
        ...     if ctx.iteration < 10:
        ...         # Use simple preconditioner early
        ...         return r / np.diag(ctx.matrix)
        ...     else:
        ...         # Use expensive preconditioner later
        ...         return neural_network(r, ctx.solution)
    """

    iteration: int
    """Current iteration number (0-indexed)."""

    residual: torch.Tensor
    """Current residual vector r_k = b - A*x_k."""

    solution: torch.Tensor
    """Current solution vector x_k."""

    matrix: torch.Tensor
    """System matrix A (optional, may be expensive to store)."""

    rhs: torch.Tensor
    """Right-hand side vector b."""


@dataclass(frozen=True, slots=True)
class PlotPaths:
    """Paths to generated diagnostic plots from a comparison workflow.

    Args:
        convergence: Convergence curve plot.
        condition_numbers: Condition number bar chart.
        parity: Parity plot (predicted vs true).
        residuals: Residual history plot.
        iterations_barplot: Horizontal bar chart of iteration counts.
        error_convergence: Relative energy-norm error convergence plot.
    """

    convergence: Path | None = None
    condition_numbers: Path | None = None
    parity: Path | None = None
    residuals: Path | None = None
    iterations_barplot: Path | None = None
    error_convergence: Path | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Path] | None) -> PlotPaths:
        """Build typed plot paths from a loose mapping.

        Args:
            mapping: Dict of plot type → path.

        Returns:
            PlotPaths with matched fields populated.
        """
        if mapping is None:
            return cls()
        return cls(
            convergence=mapping.get("convergence"),
            condition_numbers=mapping.get("condition_numbers"),
            parity=mapping.get("parity"),
            residuals=mapping.get("residuals"),
            iterations_barplot=mapping.get("iterations_barplot"),
            error_convergence=mapping.get("error_convergence"),
        )

    def to_mapping(self) -> dict[str, Path]:
        """Return only populated plot paths.

        Returns:
            Dict of plot type → path (omits None entries).
        """
        values = {
            "convergence": self.convergence,
            "condition_numbers": self.condition_numbers,
            "parity": self.parity,
            "residuals": self.residuals,
            "iterations_barplot": self.iterations_barplot,
            "error_convergence": self.error_convergence,
        }
        return {key: path for key, path in values.items() if path is not None}


@dataclass(frozen=True, slots=True)
class RankedRecommendation:
    """A single ranked entry from a solver comparison.

    Args:
        label: Preconditioner/solver label.
        iterations: Iterations to converge.
        residual: Final relative residual.
        residual_abs: Final absolute residual.
        breakdown: Whether breakdown occurred.
    """

    label: str
    iterations: int
    residual: float
    residual_abs: float
    breakdown: bool


@dataclass(frozen=True, slots=True)
class ComparisonRecommendations:
    """Ranked recommendations from a comparison run.

    Args:
        ranked: All converged entries sorted by residual (best first).
        overall_best: The single best entry, or None if none converged.
    """

    ranked: tuple[RankedRecommendation, ...] = ()
    overall_best: RankedRecommendation | None = None


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Typed result from compare_preconditioners workflow.

    Args:
        results: Raw CG comparison results keyed by preconditioner name.
        summary: Formatted text summary of results.
        solver_params: Solver parameters and data context used.
        plot_paths: Paths to generated diagnostic plots.
        preconditioners: Preconditioner names tested.
        condition_numbers: Condition numbers keyed by preconditioner name.
        recommendations: Ranked recommendations.
        output_dir: Root output directory for comparison artifacts.
        matrix_shape: Shape of the system matrix A, e.g. ``(n, n)``.
        rhs_shape: Shape of the right-hand side vector b, e.g. ``(n,)``.
        condition_number_raw: Condition number of the raw (unpreconditioned) matrix.
        rhs_source_kind: Provenance of the RHS (e.g. gaussian, sparse, raw_lhs,
            raw_rhs, dataset) — a matrix may be paired with several different
            RHS sources across sibling ``[[comparisons]]`` entries, each its
            own MLflow run, so this disambiguates which one a given run used.
    """

    results: dict[str, Any]
    summary: str
    solver_params: ComparisonGeneral
    plot_paths: PlotPaths = field(default_factory=PlotPaths)
    preconditioners: tuple[str, ...] = ()
    condition_numbers: dict[str, float] = field(default_factory=dict)
    recommendations: ComparisonRecommendations = field(default_factory=ComparisonRecommendations)
    output_dir: Path | None = None
    matrix_shape: tuple[int, int] | None = None
    rhs_shape: tuple[int, ...] | None = None
    condition_number_raw: float | None = None
    rhs_source_kind: ComparisonRhsSourceKind | None = None
