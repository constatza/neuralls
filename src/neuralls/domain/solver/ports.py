"""Ports owned by the solver domain for cross-cutting comparison concerns.

Kept separate from ``domain/inference_ports.py`` (a different, unrelated
port for batch inference) — this module is solver/comparison-specific.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from neuralls.domain.solver.models.result import ComparisonResult


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
