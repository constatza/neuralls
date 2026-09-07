"""Application-layer reporting models for assignment execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a single task execution (e.g., training, prediction).

    Args:
        name: Task name.
        artifacts: Paths to produced artifacts.
        metrics: Computed metrics.
        success: Whether the task succeeded.
        error: Error message if failed.
    """

    name: str
    artifacts: list[Path] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class AssignmentResult:
    """Final comprehensive report for a single assignment.

    Args:
        assignment_id: Stable assignment identifier.
        assignment_display_name: Human-readable label.
        status: "Success" or "Failed".
        tasks: Individual task outcomes.
        error: Top-level error message if failed.
        mlflow_run_id: The MLflow run this assignment's checkpoint/metrics live
            under — the reused run on a cache hit, or the freshly finalized run
            otherwise. None only when the assignment failed before any run was
            established.
    """

    assignment_id: str
    assignment_display_name: str
    status: str
    tasks: list[TaskResult] = field(default_factory=list)
    error: str | None = None
    mlflow_run_id: str | None = None

    @property
    def is_success(self) -> bool:
        """Whether the assignment succeeded."""
        return self.status == "Success"


@dataclass(frozen=True)
class AssignmentSweepResult:
    """Outcome of running every assignment in one case config as a sweep.

    Args:
        results: Per-assignment outcomes, in case-config declaration order.
        tracking_uri: MLflow tracking URI for the sweep's session parent run,
            or None if no MLflow session was established.
        parent_run_id: The sweep's session parent run id, or None if no
            MLflow session was established.
    """

    results: list[AssignmentResult]
    tracking_uri: str | None
    parent_run_id: str | None
