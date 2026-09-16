"""Pydantic models for individual data generation strategy configurations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuralls.shared.constants import (
    DEFAULT_KRYLOV_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SHUFFLE,
    EIGENVECTOR_SELECT_SMALLEST,
    MAX_ITERATIONS_UPPER_LIMIT,
    MIN_TOLERANCE,
    EigenvectorSelectionMode,
)

from .step_window import StepWindow


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

    method: Literal["direct", "cg"] = Field(
        "direct",
        description="Solving method: 'direct' (O(n³) exact) or 'cg' (O(n²*iters) iterative)",
    )
    rtol: float = Field(
        MIN_TOLERANCE,
        description="Relative tolerance for CG solver (ignored for direct solve)",
        ge=0.0,
    )
    atol: float = Field(
        0.0,
        description="Absolute tolerance for CG solver (ignored for direct solve)",
        ge=0.0,
    )
    max_iters: int = Field(
        MAX_ITERATIONS_UPPER_LIMIT,
        description="Maximum CG iterations (ignored for direct solve)",
        ge=1,
        le=MAX_ITERATIONS_UPPER_LIMIT,
    )
    assume_pos_def: bool = Field(
        True,
        description="Assume matrix is positive-definite (for direct solve assume_a='pos')",
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
    samples: int = Field(
        ...,
        description="Number of samples to generate for this strategy. (0=skip, -1=all, >0=exact count)",
        ge=-1,  # samples can be -1 for all
    )
    seed: int | None = Field(DEFAULT_RANDOM_SEED, description="Random seed for reproducibility.")
    shuffle: bool = Field(DEFAULT_SHUFFLE, description="Whether to shuffle the generated samples.")

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

    stop: int = Field(
        ...,
        description="Iteration/sweep cap — the solver/smoother runs exactly this many steps, never more.",
        ge=1,
    )
    start: int | None = Field(
        None,
        description=(
            "First step kept (inclusive). Unset or negative is relative to the "
            "trajectory's true end (unset keeps only the final step); non-negative "
            "is an absolute index from the beginning."
        ),
    )
    step: int = Field(
        1,
        description="Keep every Nth step within [start, stop].",
        ge=1,
    )

    @property
    def window(self) -> StepWindow:
        """The `StepWindow` these fields describe."""
        return StepWindow(start=self.start, stop=self.stop, step=self.step)

    @model_validator(mode="after")
    def _validate_window(self) -> _StepWindowFields:
        _ = self.window  # raises ValueError via StepWindow.__post_init__ if inconsistent
        return self


_UNREACHABLE_TOLERANCE = 1e-20
"""Below float64 machine epsilon (~2.2e-16) — the solver can never satisfy
this, so it always runs exactly `stop` iterations. The default for
`_CgTraceFields.rtol`/`atol`: forces a fixed-length trajectory (every
trajectory has exactly `stop + 1` rows) unless a real, reachable tolerance
is given, in which case the solver may stop earlier (see `StepWindow`'s
docstring on variable-length trajectories)."""


class _CgTraceFields(_StepWindowFields):
    """`_StepWindowFields` plus the CG solver's own early-stop tolerances.

    Only for strategies that actually run a CG solver (`residuals.py`,
    `search_directions.py`) — kept separate from `_StepWindowFields` itself
    since `smoother_filtered_probes` has no tolerance/convergence concept.
    """

    rtol: float = Field(
        _UNREACHABLE_TOLERANCE,
        description="CG relative convergence tolerance — the solver may stop before `stop` once satisfied.",
        gt=0.0,
    )
    atol: float = Field(
        _UNREACHABLE_TOLERANCE,
        description="CG absolute convergence tolerance — same early-stop semantics as `rtol`.",
        gt=0.0,
    )

    @model_validator(mode="after")
    def _validate_tolerance_start_combination(self) -> _CgTraceFields:
        """Reject a real tolerance combined with an absolute forward `start`.

        A non-negative `start` presumes the trajectory reaches that index;
        a real (reachable) tolerance means CG may stop before it does,
        which would silently select zero rows for that system. Rejecting
        the combination outright — rather than letting it fail
        unpredictably depending on runtime convergence — means real-
        tolerance mode is only ever combined with end-relative selection
        (`start` unset or negative), for which "however far it actually
        got" is always well-defined and never empty.
        """
        uses_real_tolerance = (
            self.rtol != _UNREACHABLE_TOLERANCE or self.atol != _UNREACHABLE_TOLERANCE
        )
        if uses_real_tolerance and self.start is not None and self.start >= 0:
            raise ValueError(
                "a real rtol/atol (early convergence) can only be combined with "
                "start=None or a negative (end-relative) start — an absolute "
                "forward start can silently select zero rows if CG converges early"
            )
        return self


class BaseEigenvectorConfig(BaseStrategyConfig):
    which: EigenvectorSelectionMode = Field(
        EIGENVECTOR_SELECT_SMALLEST, description="Which eigenvalues to compute."
    )
    include_eigenvectors: bool = Field(
        True, description="Whether to include eigenvectors in the generated solutions."
    )
    num_eigenvectors: int = Field(1, description="Number of eigenvectors to include.", ge=1)


class EigenvectorForwardConfig(BaseEigenvectorConfig):
    """Configuration for EigenvectorForwardStrategy."""


class EigenvectorInverseConfig(BaseEigenvectorConfig):
    """Configuration for EigenvectorInverseStrategy.

    Generates RHS vectors from eigenvector combinations, then solves for solutions.
    Supports both direct solve (O(n³)) and CG solve (O(n² * iters)).
    """

    solve_config: SolveConfig = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class KrylovConfig(BaseStrategyConfig):
    krylov_iters: int = Field(
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
    (`start` unset) only the fully-damped final sweep is kept, matching a
    single Jacobi-damping call; pass `start` to harvest multiple sweep
    depths per probe instead.
    """

    samples: int = Field(
        ...,
        description="Number of probes to generate. (>0=exact count)",
        ge=1,  # narrower than BaseStrategyConfig's ge=-1: no archive/finite
        # source exists for this strategy, so samples=-1 ("all") is never
        # meaningful — reject it at construction, not deep inside generate().
    )
    omega: float = Field(
        0.67,
        description=(
            "Weighted-Jacobi damping factor, matching "
            "torchalg.preconditioners.implementations.amg.smoothers.JacobiSmoother's default."
        ),
        gt=0.0,
    )
    probe_distribution: Literal["gaussian", "rademacher"] = Field(
        "gaussian",
        description="Distribution the initial, unfiltered probe vectors are drawn from.",
    )


class RandomNormalConfig(BaseStrategyConfig):
    target_rhs_scale: float = Field(
        1.0,
        description="Target scale for the generated RHS vectors (Euclidean norm).",
        gt=0.0,
    )


class GaussianForwardConfig(BaseStrategyConfig):
    """Configuration for GaussianForwardStrategy."""

    mu: float = Field(0.0, description="Mean of the Gaussian distribution.")
    sigma: float = Field(
        1.0,
        description="Standard deviation of the Gaussian distribution.",
        gt=0.0,
    )


class GaussianInverseConfig(GaussianForwardConfig):
    """Configuration for GaussianInverseStrategy."""

    solve_config: SolveConfig = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class UniformForwardConfig(BaseStrategyConfig):
    """Configuration for UniformForwardStrategy."""

    a: float = Field(0.0, description="Lower bound of the uniform distribution.")
    b: float = Field(1.0, description="Upper bound of the uniform distribution.")


class UniformInverseConfig(UniformForwardConfig):
    """Configuration for UniformInverseStrategy."""

    solve_config: SolveConfig = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class ConstantForwardConfig(BaseStrategyConfig):
    """Configuration for ConstantForwardStrategy."""

    c: float = Field(1.0, description="Constant value for all generated entries.")


class ConstantInverseConfig(ConstantForwardConfig):
    """Configuration for ConstantInverseStrategy."""

    solve_config: SolveConfig = Field(
        default_factory=_default_direct_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class BaseTraceConfig(BaseStrategyConfig, _CgTraceFields):
    """Shared configuration for archive-backed CG trace-collection strategies.

    By default (`start` unset) only the final CG iterate is kept; pass
    `start=0` (with `step=1`) to reproduce a full 0..stop trace.
    """

    solutions_glob: str | None = Field(
        None,
        description="Glob pattern for archive solution files to seed generation.",
    )
    archive_solutions: bool = Field(
        False,
        description="Whether to archive intermediate solutions for each iteration step.",
    )
    archive_rhs: bool = Field(
        False,
        description="Whether to archive intermediate RHS vectors for each iteration step.",
    )


class ResidualErrorConfig(BaseTraceConfig):
    """Configuration for residual-error trace strategies."""


class SearchDirectionsConfig(BaseStrategyConfig, _CgTraceFields):
    """Configuration for SearchDirectionsStrategy.

    Collects (A @ p_k, p_k) pairs from CG iterations for training neural preconditioners.
    Training mapping: NN(A @ p_k) ≈ p_k, so NN ≈ A^{-1}

    Sibling of `BaseTraceConfig`, not a subclass — this strategy has no
    archive-backed solution/RHS source, so it deliberately does not inherit
    `solutions_glob`/`archive_solutions`/`archive_rhs`.
    """


class RhsArchiveConfig(BaseStrategyConfig):
    rhs_glob: str = Field(
        ..., description="Glob pattern for RHS files to load. Overrides source.rhs_path."
    )
    solve_config: SolveConfig = Field(
        default_factory=_default_cg_solve_config,
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


class SolutionArchiveConfig(BaseStrategyConfig):
    solutions_glob: str = Field(
        ..., description="Glob pattern for solution files to load. Overrides source.solutions_path."
    )
    skip: int = Field(
        0,
        description="Number of solution files to skip after deterministic ordering/shuffling.",
        ge=0,
    )


class ValidatedArchiveConfig(BaseStrategyConfig):
    """Configuration for ValidatedArchiveStrategy.

    Loads both solutions and RHS from archives, then verifies A @ x = b consistency.
    """

    solutions_glob: str = Field(..., description="Glob pattern for solution files to load.")
    rhs_glob: str = Field(..., description="Glob pattern for RHS files to load.")
    skip: int = Field(
        0,
        description="Number of paired archive files to skip after deterministic ordering/shuffling.",
        ge=0,
    )
    verification_tolerance: float = Field(
        1e-10,
        description="Maximum relative residual ||A@x - b|| / ||b|| to accept as valid.",
        ge=0.0,
    )
    fail_on_invalid: bool = Field(
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

    solutions_glob: str = Field(..., description="Glob pattern for solution files.")
    scale: float = Field(
        5.0,
        description="Fixed scalar multiplier applied to all (x, b) pairs.",
        gt=0.0,
    )
    skip: int = Field(
        0,
        description="Number of solution files to skip after deterministic ordering/shuffling.",
        ge=0,
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

    indices: list[int] = Field(
        ..., description="Positions of nonzero entries (negative indices supported)."
    )
    values: list[float] = Field(
        ..., description="Values at the specified positions. Must match len(indices)."
    )
    solve_config: SolveConfig = Field(
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
