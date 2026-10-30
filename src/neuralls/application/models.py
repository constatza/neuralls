"""Application-layer reporting models for assignment execution results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from neuralls.domain.presentation.run_names import TrainingPresentation

type AssignmentStatus = Literal["Success", "Failed"]


@dataclass(frozen=True, slots=True)
class TaskSucceeded:
    """A task that finished (e.g., training, prediction).

    Args:
        name: Task name.
        artifacts: Paths to produced artifacts.
        metrics: Computed metrics.
    """

    name: str
    artifacts: tuple[Path, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskFailed:
    """A task that failed.

    Args:
        name: Task name.
        error: Why it failed.
    """

    name: str
    error: str


type TaskResult = TaskSucceeded | TaskFailed
"""The closed set of outcomes of a single task execution."""


@dataclass(frozen=True, slots=True)
class AssignmentSucceeded:
    """An assignment whose checkpoint and metrics live under an MLflow run.

    Args:
        assignment_id: Stable assignment identifier.
        presentation: What the assignment trains and the notation it is described with;
            every sink renders its own text from it.
        mlflow_run_id: The run holding the checkpoint and metrics — the reused run on a
            cache hit, or the freshly finalized run otherwise.
        tasks: Individual task outcomes.
    """

    assignment_id: str
    presentation: TrainingPresentation
    mlflow_run_id: str
    tasks: tuple[TaskResult, ...] = ()

    @property
    def status(self) -> AssignmentStatus:
        """Always ``"Success"``."""
        return "Success"

    @property
    def is_success(self) -> bool:
        """Always true."""
        return True

    @property
    def error(self) -> None:
        """A success carries no error."""
        return None


@dataclass(frozen=True, slots=True)
class AssignmentFailed:
    """An assignment that failed, before or after a run was established.

    Args:
        assignment_id: Stable assignment identifier.
        presentation: What the assignment trains and the notation it is described with;
            every sink renders its own text from it.
        error: Why it failed.
        tasks: Individual task outcomes.
    """

    assignment_id: str
    presentation: TrainingPresentation
    error: str
    tasks: tuple[TaskResult, ...] = ()

    @property
    def status(self) -> AssignmentStatus:
        """Always ``"Failed"``."""
        return "Failed"

    @property
    def is_success(self) -> bool:
        """Always false."""
        return False

    @property
    def mlflow_run_id(self) -> None:
        """A failed assignment has no run whose results can be read."""
        return None


type AssignmentResult = AssignmentSucceeded | AssignmentFailed
"""The closed set of final reports for a single assignment."""


@dataclass(frozen=True, slots=True)
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
