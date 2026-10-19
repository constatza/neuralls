"""Leaf value types for uncertainty, paired comparison and break-even results.

Imports nothing from the solver models, so `result.py` can hold them in its
group outcomes without a cycle. References: percentile bootstrap intervals
(Efron & Tibshirani 1993, "An Introduction to the Bootstrap"), the Wilcoxon
signed-rank test on paired differences (Wilcoxon 1945), and the cost-per-success
view of ERT (Hansen et al. 2016, "COCO: Performance Assessment", arXiv:1605.03560).
"""

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


@dataclass(frozen=True, slots=True)
class BreakEvenAt:
    """The method becomes cheaper than the reference beyond this many solved systems."""

    n_systems: float


@dataclass(frozen=True, slots=True)
class AlwaysCheaper:
    """The method is cheaper than the reference at every number of solved systems."""


@dataclass(frozen=True, slots=True)
class NeverCheaper:
    """The method is never cheaper than the reference."""


@dataclass(frozen=True, slots=True)
class CheaperOnlyBelow:
    """The method is cheaper only for fewer solved systems than this (a lower setup, a higher rate)."""

    n_systems: float


type BreakEven = BreakEvenAt | AlwaysCheaper | NeverCheaper | CheaperOnlyBelow


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """A method against the reference on the systems both solved.

    Attributes:
        n_both_solved: Systems solved by both with comparable times.
        speedups: Per system, reference time divided by method time (above 1: the
            method was faster).
        median_speedup: Median of `speedups`.
        median_speedup_ci: Bootstrap interval of the median, `None` below two pairs.
        wilcoxon_p: Two-sided Wilcoxon signed-rank p-value of the log speedups
            against zero, `None` with too few non-zero differences.
    """

    n_both_solved: int
    speedups: tuple[float, ...]
    median_speedup: float
    median_speedup_ci: Interval | None
    wilcoxon_p: float | None


@dataclass(frozen=True, slots=True)
class MethodComparison:
    """One method's comparison with the group's reference method."""

    reference: str
    break_even: BreakEven
    paired: PairedComparison | None
