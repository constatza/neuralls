"""Explicit cost scenarios for failure-aware profiles."""

from __future__ import annotations

from collections.abc import Callable

from neuralls.domain.solver.models.outcomes import Solved
from neuralls.domain.solver.models.workload import MethodWorkloadSummary

type Cost = Callable[[Solved], float]
"""One numeric scenario for a solved observation."""


def iterations(solved: Solved) -> float:
    """Cost in iterations: the iteration count of a solved run."""
    return float(solved.iterations)


def solve_seconds(solved: Solved) -> float:
    """PCG/FCG execution time for one converged system."""
    return solved.solve_s


def amortized_online_seconds(solved: Solved) -> float:
    """Solve time plus the system's planned share of its matrix setup."""
    return solved.amortized_online_s


type WorkloadTotal = Callable[[MethodWorkloadSummary], float]
"""One numeric total over a method's whole workload, not a per-system share."""


def solve_total_seconds(summary: MethodWorkloadSummary) -> float:
    """Solve seconds summed over every attempted system."""
    return summary.solve_total_s


def workload_online_seconds(summary: MethodWorkloadSummary) -> float:
    """Matrix setups plus all solves; the per-system share is `amortized_online_seconds`."""
    return summary.workload_online_s
