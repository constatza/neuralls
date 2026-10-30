"""Ports owned by the solver domain for cross-cutting comparison concerns.

Kept separate from ``domain/inference_ports.py`` (a different, unrelated
port for batch inference) — this module is solver/comparison-specific.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from neuralls.domain.solver.models.workload import MethodWorkload, MethodWorkloadSummary


@runtime_checkable
class BenchmarkMethodSink(Protocol):
    """Persist one completed method workload after it has been reduced.

    The batch owns MLflow run lifetimes.  Implementations write to the active
    method child but never create a run themselves; this keeps raw event
    emission free of aggregate metric side effects and makes the method the
    smallest reusable benchmark unit.
    """

    def record_method(
        self,
        workload: MethodWorkload,
        summary: MethodWorkloadSummary,
    ) -> None:
        """Persist canonical events and selected post-aggregation scalars."""
        ...
