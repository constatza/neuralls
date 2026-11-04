"""Leaf value types for uncertainty and iteration statistics."""

from __future__ import annotations

from dataclasses import dataclass


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
