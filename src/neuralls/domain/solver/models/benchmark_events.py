"""Canonical raw benchmark setup and solve events."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from neuralls.shared.rhs_recipe import RhsRecipe
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId


@dataclass(frozen=True, slots=True)
class MeasuredPeakMemory:
    """Peak memory observed during one timed operation."""

    bytes: int

    def __post_init__(self) -> None:
        """Reject impossible memory observations at the measurement boundary."""
        if self.bytes < 0:
            raise ValueError("peak memory bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class UnknownPeakMemory:
    """Peak memory was not available for this operation."""


type PeakMemory = MeasuredPeakMemory | UnknownPeakMemory
"""The measured or explicitly unknown memory state of one operation."""


@dataclass(frozen=True, slots=True)
class SetupCompleted:
    """A preconditioner setup finished successfully."""


@dataclass(frozen=True, slots=True)
class SetupFailed:
    """A preconditioner setup failed with its observed reason."""

    reason: str


type SetupOutcome = SetupCompleted | SetupFailed
"""The closed outcome set for one preconditioner setup."""


@dataclass(frozen=True, slots=True)
class SolveConverged:
    """A solve reached the target tolerance."""

    iterations: int
    final_relative_residual: float

    def __post_init__(self) -> None:
        """Reject impossible completed-solve measurements."""
        _validate_non_negative(self.iterations, "iterations")
        _validate_non_negative(self.final_relative_residual, "final relative residual")


@dataclass(frozen=True, slots=True)
class SolveFailed:
    """A solve stopped without reaching its target tolerance."""

    iterations: int
    reason: str

    def __post_init__(self) -> None:
        """Reject impossible failed-solve measurements."""
        _validate_non_negative(self.iterations, "iterations")


type SolveOutcome = SolveConverged | SolveFailed
"""The closed outcome set for one solve attempt."""


@dataclass(frozen=True, slots=True)
class SystemTag:
    """Where one planned system sits in a dataset, recipe, matrix, and RHS ordinal."""

    dataset: DatasetId
    recipe: RhsRecipe
    matrix_id: MatrixId
    rhs_ordinal: int
    system_id: SystemId

    def __post_init__(self) -> None:
        """Reject an impossible RHS ordinal before report grouping."""
        _validate_non_negative(self.rhs_ordinal, "RHS ordinal")


@dataclass(frozen=True, slots=True)
class MatrixSetupEvent:
    """The raw timing and outcome of setting one method up for one matrix."""

    event_id: SetupEventId
    dataset: DatasetId
    matrix_id: MatrixId
    method: PreconditionerName
    setup_s: float
    samples_s: tuple[float, ...]
    peak_memory: PeakMemory
    outcome: SetupOutcome

    def __post_init__(self) -> None:
        """Reject impossible setup durations before they enter benchmark summaries."""
        _validate_non_negative(self.setup_s, "setup time")
        _validate_samples(self.samples_s)


@dataclass(frozen=True, slots=True)
class SolveAttempt:
    """The raw timing and outcome of one method's solve attempt for one system."""

    setup_event_id: SetupEventId
    system_id: SystemId
    dataset: DatasetId
    recipe: RhsRecipe
    matrix_id: MatrixId
    rhs_ordinal: int
    method: PreconditionerName
    solve_s: float
    samples_s: tuple[float, ...]
    peak_memory: PeakMemory
    outcome: SolveOutcome

    def __post_init__(self) -> None:
        """Reject impossible solve timings and RHS ordinals before aggregation."""
        _validate_non_negative(self.rhs_ordinal, "RHS ordinal")
        _validate_non_negative(self.solve_s, "solve time")
        _validate_samples(self.samples_s)


def _validate_samples(samples_s: tuple[float, ...]) -> None:
    """Validate every raw timing sample once at the domain boundary."""
    for sample_s in samples_s:
        _validate_non_negative(sample_s, "timing sample")


def _validate_non_negative(value: float, name: str) -> None:
    """Reject quantities that cannot represent a physical benchmark measurement."""
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
