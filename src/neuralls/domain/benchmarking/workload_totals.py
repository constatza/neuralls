"""Totals over a method's whole workload, the quantities the sum-bar figures draw."""

from __future__ import annotations

from collections.abc import Callable

from neuralls.domain.solver.models.workload import MethodWorkloadSummary

type WorkloadTotal = Callable[[MethodWorkloadSummary], float]
"""One numeric total over a method's whole workload, not a per-system share."""


def solve_total_seconds(summary: MethodWorkloadSummary) -> float:
    """Solve seconds summed over every attempted system."""
    return summary.solve_total_s


def workload_online_seconds(summary: MethodWorkloadSummary) -> float:
    """Matrix setups plus all solves, the total the online-time bars draw."""
    return summary.workload_online_s
