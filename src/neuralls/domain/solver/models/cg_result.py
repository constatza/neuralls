"""Result of one CG/PCG/FCG solve in a comparison experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neuralls.domain.solver.models.benchmark_events import (
    ExactReference,
    ExactSolutionUnavailable,
    ResidualMeasurement,
    UnmeasuredResidual,
)
from neuralls.domain.solver.models.cost import (
    NO_OFFLINE_COST,
    OfflineStages,
    Unavailable,
    recorded_seconds,
)
from neuralls.shared.device import RepeatAgreement, SingleRun


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
        offline_cost: Dataset-generation and training cost behind this preconditioner's
            artifact; nothing is charged when the method has no such stage.
        setup_cost: Representative setup time for this method and matrix. The
            same measured event is copied to sibling RHS diagnostic results;
            summing these fields would count setup repeatedly.
        solve_time_seconds: Representative solve-only time for this RHS.
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

    true_residual: ResidualMeasurement = field(default_factory=UnmeasuredResidual)
    """True relative residual ``||b - A x|| / ||b||`` of the returned solution."""

    exact_reference: ExactReference = field(
        default_factory=lambda: ExactSolutionUnavailable("not computed")
    )
    """Outcome of the reference solve behind ``exact_error`` and the exact energy curve."""

    timing_agreement: RepeatAgreement = field(default_factory=SingleRun)
    """Whether the timed solve repeats reproduced the first one's iteration count."""

    offline_cost: OfflineStages = NO_OFFLINE_COST
    """Dataset-generation and training cost behind this preconditioner's artifact. The solver
    layer knows nothing about either, so a fresh result has no offline cost; the composition
    layer replaces it with the method's resolved cost. A comparison run never generates a
    dataset or trains a model itself — those are separate, earlier pipeline stages."""

    setup_cost: float = 0.0
    """Full matrix setup time copied to this RHS diagnostic result, never an
    amortized share. Aggregate totals use the linked setup event once per matrix.
    A placeholder with no attempted build has 0.0 (see comparison_run.py::_breakdown_result)."""

    setup_peak_memory_bytes: int | None = None
    """Peak memory while constructing and `.setup()`-ing this preconditioner.
    None only for a placeholder result where no build was ever attempted."""

    solve_time_seconds: float | None = None
    """Wall time for the CG/PCG/FCG solve. None if not measured."""

    solve_peak_memory_bytes: int | None = None
    """Peak memory during the solve. None if not measured."""

    setup_time_samples: tuple[float, ...] = ()
    """Every timed build+setup execution behind `setup_cost` (its statistic), in run
    order. Empty for a cache hit (the cost was copied forward) or a placeholder."""

    solve_time_samples: tuple[float, ...] = ()
    """Every timed solve execution behind `solve_time_seconds` (its statistic), in run
    order. Empty when the solve was not measured."""

    @property
    def offline_time_seconds(self) -> float | None:
        """Historical generation plus training time, kept separate from online benchmark costs."""
        return recorded_seconds(self.offline_cost.total())

    @property
    def has_unavailable_cost(self) -> bool:
        """True when an offline stage applies but its cost couldn't be resolved to a number.

        ``setup_cost`` can no longer be unavailable — every preconditioner's
        `.setup()` call is always measured.
        """
        return isinstance(self.offline_cost.total(), Unavailable)

    @property
    def avg_iteration_time_seconds(self) -> float | None:
        """Solve time divided by iteration count — the averaged per-iteration solve cost.

        Does not include generation or setup overhead.
        """
        if self.solve_time_seconds is None or self.iterations <= 0:
            return None
        return self.solve_time_seconds / self.iterations

    @property
    def peak_memory_bytes(self) -> int | None:
        """Max of setup and solve peak memory, or None if neither was measured."""
        values = [
            v for v in (self.setup_peak_memory_bytes, self.solve_peak_memory_bytes) if v is not None
        ]
        return max(values) if values else None
