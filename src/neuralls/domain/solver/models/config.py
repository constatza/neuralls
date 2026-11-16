"""Configuration models for solver execution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from torchalg.monitoring import TraceMode

from neuralls.shared.digest import Cosmetic
from neuralls.shared.system_scaling import GershgorinScaling, RhsScaling, SystemScaling
from neuralls.shared.timing import TimingRepeatPolicy
from neuralls.shared.types import ComparisonRhsSourceKind, PreconditionerName

REFERENCE_PRECISION_MARGIN = 1e-3
"""Default reference residual target is three orders below the comparison rtol."""


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Structured configuration for solver execution.

    This dataclass captures the parameters used to run a solver,
    enabling reproducible execution and structured export.
    """

    algorithm: str
    """Name of the algorithm (e.g., 'preconditioned_cg')."""

    rtol: float
    """Relative tolerance for convergence."""

    atol: float
    """Absolute tolerance for convergence."""

    maxiter: int
    """Maximum number of iterations allowed."""

    trace_mode: TraceMode
    """Verbosity of iteration tracking."""

    m_max: int | None = None
    """Orthogonalization window size (used by FCG)."""

    preconditioner: str | None = None
    """Name or description of the preconditioner used."""

    extra_params: dict[str, Any] = field(default_factory=dict)
    """Additional algorithm-specific parameters."""


@dataclass(frozen=True, slots=True)
class SolverParams:
    """Numerical solver parameters for comparison runs.

    Args:
        rtol: Relative convergence tolerance.
        atol: Absolute convergence tolerance.
        max_iterations: Maximum iterations allowed.
        stopping_criterion: When to stop iteration.
        m_max: FCG orthogonalization restart parameter.
        reference_precision_margin: How many orders of magnitude tighter than
            ``rtol`` the host reference solution must be (see
            ``error_metrics.reference_target``). The comparison config validates it to
            ``(0, 1)``; a ``SolverParams`` built directly is trusted.
        timing: How often each setup and solve is timed.
    """

    rtol: float
    atol: float
    max_iterations: int
    stopping_criterion: Literal["residual_norm"]
    m_max: int
    reference_precision_margin: float = REFERENCE_PRECISION_MARGIN
    timing: Annotated[TimingRepeatPolicy, Cosmetic()] = field(default_factory=TimingRepeatPolicy)
    """Measurement effort only — it changes the noise of the recorded times, not what is
    computed, so `Cosmetic()`: editing it never silently invalidates (and reruns) every
    expensive prior comparison. Pass `--force` to remeasure with a higher setting."""


@dataclass(frozen=True, slots=True)
class ComparisonData:
    """Input data and scaling context for a comparison run.

    Args:
        matrix_path: Path to system matrix file.
        matrix_index: Physical storage row of the matrix to solve on (0 for a
            single-matrix dataset, whose one matrix is stored once).
        dataset_alias: Optional dataset label.
        system_scaling: How the matrix and RHS are scaled; must equal a stored dataset's.
        rhs_scaling: Per-system RHS scaling applied after ``system_scaling``.
    """

    matrix_path: Annotated[Path, Cosmetic()]
    rhs_path: Annotated[Path | None, Cosmetic()]
    matrix_index: int = 0
    dataset_alias: str | None = None
    system_scaling: SystemScaling = field(default_factory=GershgorinScaling)
    rhs_scaling: RhsScaling = RhsScaling.NONE
    require_non_residual_rhs: bool = True
    selection_seed: int | None = None
    rhs_source_kind: ComparisonRhsSourceKind | None = None
    rhs_source_params: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ComparisonGeneral:
    """Grouped comparison context: solver params + data spec.

    Args:
        params: Solver configuration.
        data: Input data and scaling spec.
    """

    params: SolverParams
    data: ComparisonData
    reference_preconditioner: PreconditionerName | None = None
