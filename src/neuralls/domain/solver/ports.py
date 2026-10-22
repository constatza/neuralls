"""Ports owned by the solver domain for cross-cutting comparison concerns.

Kept separate from ``domain/inference_ports.py`` (a different, unrelated
port for batch inference) — this module is solver/comparison-specific.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from neuralls.domain.solver.models.group_outcomes import RecipeProfileReport, WorkloadMixtureReport
from neuralls.domain.solver.models.presentation import GroupPresentation
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.domain.solver.models.workload import MethodWorkload, MethodWorkloadSummary


@runtime_checkable
class CostRecorder(Protocol):
    """Records a comparison's cost/outcome metrics somewhere durable.

    One method, matching this port's one responsibility: take a finished
    ``ComparisonResult`` (and the nested-run tags its per-preconditioner
    child runs need) and record it. The concrete implementation decides
    where — MLflow today (``platform/tracking/comparison_tracking.py``'s
    ``MLflowCostRecorder``), but composition-layer callers depend on this
    Protocol, not on MLflow directly.
    """

    def record(
        self,
        result: ComparisonResult,
        *,
        child_run_tags: Mapping[str, Mapping[str, str]],
    ) -> None:
        """Record ``result``'s cost/outcome metrics.

        Args:
            result: The finished comparison result to record.
            child_run_tags: Per-preconditioner nested-run tags, keyed by
                preconditioner name.
        """
        ...


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


class GroupReportSink(Protocol):
    """Receives one event-backed report group and records it."""

    def report(
        self,
        outcome: RecipeProfileReport | WorkloadMixtureReport,
        *,
        presentation: GroupPresentation,
    ) -> None:
        """Record the counts and inputs supported by `outcome`'s report variant."""
        ...

    def finalize(self) -> None:
        """Flush session-level artifacts while the owning run is active."""
        ...
