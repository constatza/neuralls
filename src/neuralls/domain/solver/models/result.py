"""Result dataclasses for solver comparison reporting.

This module defines result containers for comparison workflows. Per-solve
diagnostics (what torchalg calls ``SolverResult``, and the ``IterationContext``
passed to its flexible preconditioners) belong to torchalg, not here.

Design:
    - CGComparisonResult: Extended result for comparison experiments

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

from neuralls.shared.types import CostProvenance

if TYPE_CHECKING:
    from neuralls.shared.types import ComparisonRhsSourceKind

    from .config import ComparisonGeneral


@dataclass(frozen=True, slots=True)
class StageCost:
    """One pipeline stage's wall-clock cost, tagged with where the number came from.

    See shared.types.CostProvenance for what MEASURED/HISTORICAL/UNAVAILABLE mean.
    """

    wall_time_seconds: float
    peak_memory_bytes: int | None
    provenance: CostProvenance


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
        generation_cost: Dataset-generation cost feeding this preconditioner's
            fit/train step, or `None` if generation doesn't apply.
        setup_cost: Wall-clock seconds to construct and `.setup()` this
            preconditioner. Always measured — every `torchalg.Preconditioner`
            requires an explicit `setup(matrix)` call before `apply()`, so
            this cost is deterministic and present for every result.
        solve_time_seconds: Wall time for the CG/PCG/FCG solve itself.
            `None` if never measured.
        solve_peak_memory_bytes: Peak memory during the solve. `None` if
            never measured.

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
    """Preconditioner result identifier (e.g., 'identity', 'jacobi', 'neural')."""

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

    generation_cost: StageCost | None = None
    """Dataset-generation cost feeding this preconditioner's fit/train step, or None if
    generation doesn't apply (classical/geometric AMG, standard/Jacobi/IC0 preconditioners
    have no training data at all). When present it is always HISTORICAL (read from the
    dataset's manifest) or UNAVAILABLE — a comparison run never generates a dataset itself,
    that's a separate, earlier pipeline stage (see composition/generation's own docs)."""

    setup_cost: float = 0.0
    """Wall-clock seconds to construct and `.setup()` this preconditioner. Always
    measured; 0.0 only for a placeholder result where no build was ever attempted
    (see comparison_run.py::_breakdown_result)."""

    setup_peak_memory_bytes: int | None = None
    """Peak memory while constructing and `.setup()`-ing this preconditioner.
    None only for a placeholder result where no build was ever attempted."""

    solve_time_seconds: float | None = None
    """Wall time for the CG/PCG/FCG solve. None if not measured."""

    solve_peak_memory_bytes: int | None = None
    """Peak memory during the solve. None if not measured."""

    @property
    def total_time_seconds(self) -> float | None:
        """Setup + generation + solve wall time, or None if nothing could be summed.

        ``setup_cost`` is always a real measured number, so it is always
        included. ``generation_cost`` excludes an UNAVAILABLE-provenance
        component — an unavailable cost is a real, unrecorded cost, and
        counting it as 0.0 would silently understate the total.
        """
        parts = [self.setup_cost]
        if (
            self.generation_cost is not None
            and self.generation_cost.provenance is not CostProvenance.UNAVAILABLE
        ):
            parts.append(self.generation_cost.wall_time_seconds)
        if self.solve_time_seconds is not None:
            parts.append(self.solve_time_seconds)
        return sum(parts) if parts else None

    @property
    def has_unavailable_cost(self) -> bool:
        """True when generation cost exists but couldn't be resolved to a real number.

        ``setup_cost`` can no longer be UNAVAILABLE — every preconditioner's
        `.setup()` call is always measured.
        """
        return self.generation_cost is not None and (
            self.generation_cost.provenance is CostProvenance.UNAVAILABLE
        )

    @property
    def avg_iteration_time_seconds(self) -> float | None:
        """Solve time divided by iteration count — the averaged per-iteration solve cost.

        Does NOT include generation/setup overhead; see total_cost_per_iteration_seconds
        for amortized full cost.
        """
        if self.solve_time_seconds is None or self.iterations <= 0:
            return None
        return self.solve_time_seconds / self.iterations

    @property
    def total_cost_per_iteration_seconds(self) -> float | None:
        """Full pipeline cost amortized per iteration: (generation + setup + solve) / iterations.

        Includes one-time setup and generation costs spread across all iterations.
        Useful for comparing fair "cost per solve" when preconditioners have different setup costs.
        """
        total = self.total_time_seconds
        if total is None or self.iterations <= 0:
            return None
        return total / self.iterations

    @property
    def peak_memory_bytes(self) -> int | None:
        """Max of setup and solve peak memory, or None if neither was measured."""
        values = [
            v for v in (self.setup_peak_memory_bytes, self.solve_peak_memory_bytes) if v is not None
        ]
        return max(values) if values else None


@dataclass(frozen=True, slots=True)
class PlotPaths:
    """Paths to generated diagnostic plots from a comparison workflow.

    Args:
        convergence: Convergence curve plot.
        parity: Parity plot (predicted vs true).
        residuals: Residual history plot.
        iterations_barplot: Horizontal bar chart of iteration counts.
        error_convergence: Relative energy-norm error convergence plot.
        generation_time_barplot: Horizontal bar chart of dataset-generation times.
        setup_time_barplot: Horizontal bar chart of preconditioner setup times.
        solve_time_barplot: Horizontal bar chart of CG solve times.
        avg_iteration_time_barplot: Horizontal bar chart of solve time per iteration.
        total_cost_per_iteration_barplot: Horizontal bar chart of total amortized cost
            (generation + setup + solve) per iteration.
        peak_memory_barplot: Horizontal bar chart of peak memory usage.
        time_breakdown_barplot: Stacked horizontal bar chart of generation vs.
            setup vs. solve time per method.
        work_precision: Work-precision diagram (final precision vs. total cost).
        residual_vs_time: Residual vs wall time (generation + setup + solve).
        error_vs_time: Energy-norm error vs wall time (when available).
    """

    convergence: Path | None = None
    parity: Path | None = None
    residuals: Path | None = None
    iterations_barplot: Path | None = None
    error_convergence: Path | None = None
    generation_time_barplot: Path | None = None
    setup_time_barplot: Path | None = None
    solve_time_barplot: Path | None = None
    avg_iteration_time_barplot: Path | None = None
    total_cost_per_iteration_barplot: Path | None = None
    peak_memory_barplot: Path | None = None
    time_breakdown_barplot: Path | None = None
    work_precision: Path | None = None
    residual_vs_time: Path | None = None
    error_vs_time: Path | None = None

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
            parity=mapping.get("parity"),
            residuals=mapping.get("residuals"),
            iterations_barplot=mapping.get("iterations_barplot"),
            error_convergence=mapping.get("error_convergence"),
            generation_time_barplot=mapping.get("generation_time_barplot"),
            setup_time_barplot=mapping.get("setup_time_barplot"),
            solve_time_barplot=mapping.get("solve_time_barplot"),
            avg_iteration_time_barplot=mapping.get("avg_iteration_time_barplot"),
            total_cost_per_iteration_barplot=mapping.get("total_cost_per_iteration_barplot"),
            peak_memory_barplot=mapping.get("peak_memory_barplot"),
            time_breakdown_barplot=mapping.get("time_breakdown_barplot"),
            work_precision=mapping.get("work_precision"),
            residual_vs_time=mapping.get("residual_vs_time"),
            error_vs_time=mapping.get("error_vs_time"),
        )

    def to_mapping(self) -> dict[str, Path]:
        """Return only populated plot paths.

        Returns:
            Dict of plot type → path (omits None entries).
        """
        values = {
            "convergence": self.convergence,
            "parity": self.parity,
            "residuals": self.residuals,
            "iterations_barplot": self.iterations_barplot,
            "error_convergence": self.error_convergence,
            "generation_time_barplot": self.generation_time_barplot,
            "setup_time_barplot": self.setup_time_barplot,
            "solve_time_barplot": self.solve_time_barplot,
            "avg_iteration_time_barplot": self.avg_iteration_time_barplot,
            "total_cost_per_iteration_barplot": self.total_cost_per_iteration_barplot,
            "peak_memory_barplot": self.peak_memory_barplot,
            "time_breakdown_barplot": self.time_breakdown_barplot,
            "work_precision": self.work_precision,
            "residual_vs_time": self.residual_vs_time,
            "error_vs_time": self.error_vs_time,
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
        recommendations: Ranked recommendations.
        output_dir: Root output directory for comparison artifacts.
        matrix_shape: Shape of the system matrix A, e.g. ``(n, n)``.
        rhs_shape: Shape of the right-hand side vector b, e.g. ``(n,)``.
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
    recommendations: ComparisonRecommendations = field(default_factory=ComparisonRecommendations)
    output_dir: Path | None = None
    matrix_shape: tuple[int, int] | None = None
    rhs_shape: tuple[int, ...] | None = None
    rhs_source_kind: ComparisonRhsSourceKind | None = None
