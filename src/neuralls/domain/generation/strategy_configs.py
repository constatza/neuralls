"""Pydantic models for individual data generation strategy configurations."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuralls.shared.constants import (
    DEFAULT_KRYLOV_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SHUFFLE,
    MAX_ITERATIONS_UPPER_LIMIT,
    MIN_TOLERANCE,
)
from neuralls.shared.fact_binding import (
    Fact,
    Nested,
    Omitted,
    Show,
    auto_if_none,
    enum_value,
    on_off,
)
from neuralls.shared.scientific_description.catalog import (
    CONSTANT_VALUE,
    DAMPING,
    EIGENVALUE_SELECTION,
    EIGENVECTOR_COUNT,
    EIGENVECTOR_INCLUSION,
    FREQUENCY,
    ITERATION_COUNT,
    KRYLOV_DIMENSION,
    LOWER_BOUND,
    MEAN,
    PROBE_DISTRIBUTION,
    RHS_SCALE,
    SOLUTION_SCALE,
    SOLVER_ABSOLUTE_TOLERANCE,
    SOLVER_KIND,
    SOLVER_MAX_ITERATIONS,
    SOLVER_POSITIVE_DEFINITE,
    SOLVER_RELATIVE_TOLERANCE,
    SPARSE_SUPPORT,
    STANDARD_DEVIATION,
    STEPS,
    UPPER_BOUND,
    VERIFICATION_TOLERANCE,
    WINDOW_START,
    WINDOW_STRIDE,
    FactRole,
)
from neuralls.shared.types import EigenvectorSelection, MatrixFormat

from .step_window import StepWindow

_OPERATIONAL = Omitted(FactRole.OPERATIONAL)
_PROVENANCE = Omitted(FactRole.PROVENANCE)
_REPRODUCIBILITY = Omitted(FactRole.REPRODUCIBILITY)


def _is_cg(solve: SolveConfig) -> bool:
    """Tolerances and the iteration cap only parameterize the iterative solver."""
    return solve.method == "cg"


class SolveConfig(BaseModel):
    """Configuration for solving linear systems.

    Provides unified solving configuration across all inverse strategies
    (strategies that solve x = A^-1 @ b instead of computing b = A @ x).

    Attributes:
        method: Solving method ("direct" or "cg")
            - "direct": Uses scipy.linalg.solve (O(n³), exact modulo floating point)
            - "cg": Uses scipy.sparse.linalg.cg (O(n² * iters), iterative)
        rtol: Relative tolerance for CG solver (ignored for direct)
        atol: Absolute tolerance for CG solver (ignored for direct)
        max_iters: Maximum CG iterations (ignored for direct)
        assume_pos_def: Assume matrix is positive-definite (direct solve only)

    Examples:
        >>> # Direct solve (default)
        >>> SolveConfig(method="direct", assume_pos_def=True)

        >>> # CG solve with custom tolerances
        >>> SolveConfig(method="cg", rtol=1e-10, atol=1e-12, max_iters=1000)
    """

    method: Annotated[Literal["direct", "cg"], Fact(SOLVER_KIND)] = Field(
        "direct",
        description="Solving method: 'direct' (O(n³) exact) or 'cg' (O(n²*iters) iterative)",
    )
    rtol: Annotated[float, Fact(SOLVER_RELATIVE_TOLERANCE, when=_is_cg)] = Field(
        MIN_TOLERANCE,
        description="Relative tolerance for CG solver (ignored for direct solve)",
        ge=0.0,
    )
    atol: Annotated[float, Fact(SOLVER_ABSOLUTE_TOLERANCE, when=_is_cg)] = Field(
        0.0,
        description="Absolute tolerance for CG solver (ignored for direct solve)",
        ge=0.0,
    )
    max_iters: Annotated[int, Fact(SOLVER_MAX_ITERATIONS, when=_is_cg)] = Field(
        MAX_ITERATIONS_UPPER_LIMIT,
        description="Maximum CG iterations (ignored for direct solve)",
        ge=1,
        le=MAX_ITERATIONS_UPPER_LIMIT,
    )
    assume_pos_def: Annotated[bool, Fact(SOLVER_POSITIVE_DEFINITE, convert=on_off, when=_is_cg)] = (
        Field(
            True,
            description="Assume matrix is positive-definite (for direct solve assume_a='pos')",
        )
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


def _default_direct_solve_config() -> SolveConfig:
    """Build the standard direct-solve configuration."""
    return SolveConfig.model_validate(
        {
            "method": "direct",
            "rtol": MIN_TOLERANCE,
            "atol": 0.0,
            "max_iters": MAX_ITERATIONS_UPPER_LIMIT,
            "assume_pos_def": True,
        }
    )


def _default_cg_solve_config() -> SolveConfig:
    """Build the standard CG solve configuration."""
    return SolveConfig.model_validate(
        {
            "method": "cg",
            "rtol": MIN_TOLERANCE,
            "atol": 0.0,
            "max_iters": MAX_ITERATIONS_UPPER_LIMIT,
            "assume_pos_def": True,
        }
    )


class BaseStrategyConfig(BaseModel):
    samples: Annotated[int, _OPERATIONAL] = Field(
        ...,
        description="Number of samples to generate for this strategy. (0=skip, -1=all, >0=exact count)",
        ge=-1,  # samples can be -1 for all
    )
    seed: Annotated[int | None, _REPRODUCIBILITY] = Field(
        DEFAULT_RANDOM_SEED, description="Random seed for reproducibility."
    )
    shuffle: Annotated[bool, _OPERATIONAL] = Field(
        DEFAULT_SHUFFLE, description="Whether to shuffle the generated samples."
    )

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields to catch typos
        frozen=True,  # Make configs immutable
    )


class _StepWindowFields(BaseModel):
    """start/stop/step fields for any strategy harvesting a bounded trajectory.

    Mixed into a strategy config alongside `BaseStrategyConfig` (multiple
    inheritance) rather than folded into it directly, since not every
    strategy harvests a trajectory — see `StepWindow` for the semantics.
    """

    stop: Annotated[int, Fact(ITERATION_COUNT)] = Field(
        ...,
        description="Iteration/sweep cap — the solver/smoother runs exactly this many steps, never more.",
        ge=1,
    )
    start: Annotated[int | None, Fact(WINDOW_START, show=Show.NOT_NONE)] = Field(
        None,
        description=(
            "First step kept (inclusive). Omitted (None) means -1: keep only the last "
            "step. Negative values count back from the last step (-(stop+1)..-1); positive "
            "values are absolute and must be below stop (1..stop-1). The first kept step "
            "must be >= 1, since step 0 is the base pair (r0 = b, e0 = x*) and is never "
            "emitted."
        ),
    )
    step: Annotated[int, Fact(WINDOW_STRIDE, show=Show.NON_DEFAULT)] = Field(
        1,
        description="Keep every Nth step within [start, stop].",
        ge=1,
    )

    model_config = ConfigDict(frozen=True)

    @property
    def window(self) -> StepWindow:
        """The `StepWindow` these fields describe."""
        return StepWindow(start=self.start, stop=self.stop, step=self.step)

    @model_validator(mode="after")
    def _validate_window(self) -> _StepWindowFields:
        """Check the StepWindow bounds, then that the first kept step is >= 1.

        Step 0 would emit the base pair (r0 = b, e0 = x*), which generation
        never needs: CG always begins at x0 = 0, so iterate 0 is that pair.
        Negative starts resolve against the trajectory end, so the check runs
        on the indices a full-length trajectory would keep, which catches a
        start that reaches index 0 there.
        """
        window = self.window  # raises ValueError via StepWindow.__post_init__ if inconsistent
        kept = window.resolve_indices(self.stop + 1)
        if not kept:
            raise ValueError("window keeps no step")
        if kept[0] < 1:
            raise ValueError(
                f"start must keep a first step >= 1 (got first kept step {kept[0]}): "
                "step 0 is the base pair (r0 = b, e0 = x*) and is never emitted"
            )
        return self


class NeutralOnesConfig(BaseStrategyConfig):
    """Configuration for NeutralOnesStrategy.

    Generates neutral test cases where x = ones and b = A @ x.
    This provides an unbiased baseline for solver comparison across experiments.
    It has no parameters beyond the base ones.
    """


class BaseEigenvectorConfig(BaseStrategyConfig):
    which: Annotated[EigenvectorSelection, Fact(EIGENVALUE_SELECTION, convert=enum_value)] = Field(
        EigenvectorSelection.SMALLEST, description="Which eigenvalues to compute."
    )
    num_eigenvectors: Annotated[int, Fact(EIGENVECTOR_COUNT)] = Field(
        1, description="Number of eigenvectors to include.", ge=1
    )
    include_eigenvectors: Annotated[bool, Fact(EIGENVECTOR_INCLUSION, convert=on_off)] = Field(
        True, description="Whether to include eigenvectors in the generated solutions."
    )


class EigenvectorForwardConfig(BaseEigenvectorConfig):
    """Configuration for EigenvectorForwardStrategy."""


class EigenvectorInverseConfig(BaseEigenvectorConfig):
    """Configuration for EigenvectorInverseStrategy.

    Generates RHS vectors from eigenvector combinations, then solves for solutions.
    Supports both direct solve (O(n³)) and CG solve (O(n² * iters)).
    """

    solve_config: Annotated[SolveConfig, Nested()] = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class KrylovConfig(BaseStrategyConfig):
    krylov_iters: Annotated[int, Fact(KRYLOV_DIMENSION)] = Field(
        DEFAULT_KRYLOV_ITERATIONS,
        description="Number of Krylov iterations to perform.",
        ge=1,
    )


class SmootherFilteredProbesConfig(BaseStrategyConfig, _StepWindowFields):
    """Configuration for SmootherFilteredProbesStrategy.

    Random probe vectors are damped by `stop` weighted-Jacobi sweeps before
    being kept (per `window`) as error snapshots — the directions that
    survive are, by construction, the ones a Jacobi smoother handles poorly
    (what a multigrid coarse-grid correction needs to cover). By default
    (`start` omitted, i.e. -1) only the fully-damped final sweep is kept; an
    explicit `start` selects which damping sweeps are kept.
    """

    stop: Annotated[int, Fact(STEPS)] = Field(
        ...,
        description="Iteration/sweep cap — the solver/smoother runs exactly this many steps, never more.",
        ge=1,
    )
    samples: Annotated[int, _OPERATIONAL] = Field(
        ...,
        description="Number of probes to generate. (>0=exact count)",
        ge=1,  # narrower than BaseStrategyConfig's ge=-1: no archive/finite
        # source exists for this strategy, so samples=-1 ("all") is never
        # meaningful — reject it at construction, not deep inside generate().
    )
    probe_distribution: Annotated[Literal["gaussian", "rademacher"], Fact(PROBE_DISTRIBUTION)] = (
        Field(
            "gaussian",
            description="Distribution the initial, unfiltered probe vectors are drawn from.",
        )
    )
    omega: Annotated[float | None, Fact(DAMPING, convert=auto_if_none)] = Field(
        None,
        description=(
            "Weighted-Jacobi damping factor; auto per-matrix via torchalg's "
            "spectral-radius rule when unset, matching "
            "torchalg.preconditioners.implementations.amg.smoothers.JacobiSmoother's default."
        ),
        gt=0.0,
    )


class RandomNormalConfig(BaseStrategyConfig):
    target_rhs_scale: Annotated[float, Fact(RHS_SCALE)] = Field(
        1.0,
        description="Target scale for the generated RHS vectors (Euclidean norm).",
        gt=0.0,
    )


class GaussianForwardConfig(BaseStrategyConfig):
    """Configuration for GaussianForwardStrategy."""

    mu: Annotated[float, Fact(MEAN)] = Field(0.0, description="Mean of the Gaussian distribution.")
    sigma: Annotated[float, Fact(STANDARD_DEVIATION)] = Field(
        1.0,
        description="Standard deviation of the Gaussian distribution.",
        gt=0.0,
    )


class GaussianInverseConfig(GaussianForwardConfig):
    """Configuration for GaussianInverseStrategy."""

    solve_config: Annotated[SolveConfig, Nested()] = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class UniformForwardConfig(BaseStrategyConfig):
    """Configuration for UniformForwardStrategy."""

    a: Annotated[float, Fact(LOWER_BOUND)] = Field(
        0.0, description="Lower bound of the uniform distribution."
    )
    b: Annotated[float, Fact(UPPER_BOUND)] = Field(
        1.0, description="Upper bound of the uniform distribution."
    )


class UniformInverseConfig(UniformForwardConfig):
    """Configuration for UniformInverseStrategy."""

    solve_config: Annotated[SolveConfig, Nested()] = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class SinusoidalFieldConfig(BaseStrategyConfig):
    """Configuration for SinusoidalFieldStrategy."""

    omega: Annotated[float, Fact(FREQUENCY)] = Field(
        math.tau,
        description=(
            "Total phase (radians) swept across the vector in sin(omega/n * i + phi), "
            "e.g. 2*pi*k for k full cycles — independent of the vector length n."
        ),
        gt=0,
    )
    solve_config: Annotated[SolveConfig, Nested()] = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class ConstantForwardConfig(BaseStrategyConfig):
    """Configuration for ConstantForwardStrategy."""

    c: Annotated[float, Fact(CONSTANT_VALUE)] = Field(
        1.0, description="Constant value for all generated entries."
    )


class ConstantInverseConfig(ConstantForwardConfig):
    """Configuration for ConstantInverseStrategy."""

    solve_config: Annotated[SolveConfig, Nested()] = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class BaseTraceConfig(BaseStrategyConfig, _StepWindowFields):
    """Shared configuration for archive-backed CG trace-collection strategies.

    The first kept step must be >= 1 (see `_StepWindowFields`), so the base
    pair at iterate 0 is never emitted. The trajectory always runs to `stop`;
    the CG tolerance is not configurable here because any real tolerance
    would let CG stop early and break the fixed-length trajectory contract.
    """

    solutions_glob: Annotated[str | None, _PROVENANCE] = Field(
        None,
        description="Glob pattern for archive solution files to seed generation.",
    )
    archive_solutions: Annotated[bool, _OPERATIONAL] = Field(
        False,
        description="Whether to archive intermediate solutions for each iteration step.",
    )
    archive_rhs: Annotated[bool, _OPERATIONAL] = Field(
        False,
        description="Whether to archive intermediate RHS vectors for each iteration step.",
    )


class ResidualErrorConfig(BaseTraceConfig):
    """Configuration for residual-error trace strategies."""

    file_indices: Annotated[tuple[int, ...] | None, _OPERATIONAL] = Field(
        None,
        description=(
            "Explicit positions in the solutions pool, set per binding by the orchestrator "
            "so bindings draw disjoint base systems. Must match the resolved base-system count."
        ),
    )


class RhsArchiveConfig(BaseStrategyConfig):
    rhs_glob: Annotated[str, _PROVENANCE] = Field(
        ..., description="Glob pattern for RHS files to load. Overrides source.rhs_path."
    )
    solve_config: Annotated[SolveConfig, Nested()] = Field(
        default_factory=_default_cg_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )
    file_indices: Annotated[tuple[int, ...] | None, _OPERATIONAL] = Field(
        None,
        description=(
            "Explicit positions in the archive pool, set per binding by the orchestrator "
            "so bindings draw disjoint files. Must match the resolved sample count."
        ),
    )


class SolutionArchiveConfig(BaseStrategyConfig):
    solutions_glob: Annotated[str, _PROVENANCE] = Field(
        ..., description="Glob pattern for solution files to load. Overrides source.solutions_path."
    )
    skip: Annotated[int, _OPERATIONAL] = Field(
        0,
        description="Number of solution files to skip after deterministic ordering/shuffling.",
        ge=0,
    )
    file_indices: Annotated[tuple[int, ...] | None, _OPERATIONAL] = Field(
        None,
        description=(
            "Explicit positions in the archive pool, set per binding by the orchestrator "
            "so bindings draw disjoint files. Must match the resolved sample count."
        ),
    )


class ValidatedArchiveConfig(BaseStrategyConfig):
    """Configuration for ValidatedArchiveStrategy.

    Loads both solutions and RHS from archives, then verifies A @ x = b consistency.
    """

    solutions_glob: Annotated[str, _PROVENANCE] = Field(
        ..., description="Glob pattern for solution files to load."
    )
    rhs_glob: Annotated[str, _PROVENANCE] = Field(
        ..., description="Glob pattern for RHS files to load."
    )
    skip: Annotated[int, _OPERATIONAL] = Field(
        0,
        description="Number of paired archive files to skip after deterministic ordering/shuffling.",
        ge=0,
    )
    file_indices: Annotated[tuple[int, ...] | None, _OPERATIONAL] = Field(
        None,
        description=(
            "Explicit positions in the archive pool, set per binding by the orchestrator "
            "so bindings draw disjoint files. Must match the resolved sample count."
        ),
    )
    verification_tolerance: Annotated[float, Fact(VERIFICATION_TOLERANCE)] = Field(
        1e-10,
        description="Maximum relative residual ||A@x - b|| / ||b|| to accept as valid.",
        ge=0.0,
    )
    fail_on_invalid: Annotated[bool, _OPERATIONAL] = Field(
        True,
        description="Whether to raise error if any pair fails verification.",
    )


class ScaledSolutionsConfig(BaseStrategyConfig):
    """Configuration for ScaledSolutionsStrategy.

    Loads solution vectors from archive, computes b = A @ x, then scales
    both sides by a fixed scalar: returns (scale * b, scale * x).

    Attributes:
        solutions_glob: Glob pattern for solution files.
        scale: Fixed scalar multiplier applied to all (x, b) pairs.
        skip: Number of solution files to skip after deterministic ordering/shuffling.
    """

    solutions_glob: Annotated[str, _PROVENANCE] = Field(
        ..., description="Glob pattern for solution files."
    )
    scale: Annotated[float, Fact(SOLUTION_SCALE)] = Field(
        5.0,
        description="Fixed scalar multiplier applied to all (x, b) pairs.",
        gt=0.0,
    )
    skip: Annotated[int, _OPERATIONAL] = Field(
        0,
        description="Number of solution files to skip after deterministic ordering/shuffling.",
        ge=0,
    )
    file_indices: Annotated[tuple[int, ...] | None, _OPERATIONAL] = Field(
        None,
        description=(
            "Explicit positions in the archive pool, set per binding by the orchestrator "
            "so bindings draw disjoint files. Must match the resolved sample count."
        ),
    )


class SparseRhsConfig(BaseStrategyConfig):
    """Configuration for SparseRhsStrategy.

    Generates sparse RHS vectors with nonzero entries at specified indices,
    then solves x = A^-1 @ b for each system.

    Attributes:
        indices: Positions of nonzero entries (negative indices supported, e.g. [-1]).
        values: Values at the specified positions. Must match len(indices).
        solve_config: Configuration for solving the linear system.
    """

    indices: Annotated[list[int], Fact(SPARSE_SUPPORT, convert=len)] = Field(
        ..., description="Positions of nonzero entries (negative indices supported)."
    )
    values: Annotated[list[float], _PROVENANCE] = Field(
        ..., description="Values at the specified positions. Must match len(indices)."
    )
    solve_config: Annotated[SolveConfig, Nested()] = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )

    @model_validator(mode="after")
    def _validate_lengths(self) -> SparseRhsConfig:
        """Validate that indices and values have the same length."""
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"indices and values must have the same length, "
                f"got {len(self.indices)} indices and {len(self.values)} values"
            )
        return self


class MixedStrategyConfig(BaseModel):
    """Configuration for a mix of strategies. (Not directly used by `generate` methods)."""


class GenerationConfig(BaseModel):
    """Overall configuration for data generation."""

    # This represents the structure of the [generation] section in the config
    # Add fields as needed to match the actual data generation config structure.
    # For now, it's a placeholder.


EIGENVECTOR_STRATEGY_NAMES: frozenset[str] = frozenset(
    {"eigenvector_forward", "eigenvector_inverse"}
)
"""Strategy names whose `which` selection must be checked against the matrix format."""


def require_which_supported_by_format(
    which: EigenvectorSelection,
    matrix_format: MatrixFormat,
) -> None:
    """Reject eigenvector selections the dataset's matrix format cannot serve.

    "random" selection draws from the whole spectrum, which only a dense
    eigendecomposition provides. CSR matrices are restricted to the sparse
    "smallest"/"largest" eigensolve.

    Raises:
        ValueError: If `which` is "random" for a CSR dataset.
    """
    if matrix_format is MatrixFormat.CSR and which == EigenvectorSelection.RANDOM:
        raise ValueError(
            "eigenvector which='random' requires the full spectrum and is not supported "
            "with matrix_format='csr'; use 'smallest' or 'largest', or matrix_format='dense'"
        )
