"""Explicit cost scenarios for failure-aware profiles."""

from __future__ import annotations

from collections.abc import Callable

from neuralls.domain.solver.models.outcomes import Solved

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
