"""Leaf value types for uncertainty and paired comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.shared.types import PreconditionerName


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval around a point estimate.

    Attributes:
        low: Lower bound.
        high: Upper bound.
        confidence: Coverage level, e.g. 0.95.
    """

    low: float
    high: float
    confidence: float


@dataclass(frozen=True, slots=True)
class IterationStats:
    """Iteration counts over a method's converged runs only (failed runs hit the cap).

    Attributes:
        mean: Mean iterations.
        median: Median iterations.
        std: Sample standard deviation (ddof=1), `None` below two runs.
    """

    mean: float
    median: float
    std: float | None


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """A method against the reference on the systems both solved.

    Attributes:
        speed_ratios: Per system, reference time divided by method time (above 1: the
            method was faster).
        median_speed_ratio: Median ratio, or `None` without a both-solved pair.
        median_speed_ratio_ci: Matrix-cluster interval, `None` below two matrices.
        n_both_solved: Systems solved by both.
        n_reference_only: Systems solved only by the reference.
        n_method_only: Systems solved only by the method.
    """

    speed_ratios: tuple[float, ...]
    median_speed_ratio: float | None
    median_speed_ratio_ci: Interval | None
    n_both_solved: int
    n_reference_only: int
    n_method_only: int


@dataclass(frozen=True, slots=True)
class MethodComparison:
    """One method's comparison with the group's reference method."""

    reference: PreconditionerName
    paired: PairedComparison
