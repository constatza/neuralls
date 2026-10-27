"""Domain models for the assignment configuration system.

An assignment pairs one job with one dataset — the intent "run this job on
this dataset." These classes define the core data structures for assignments,
strictly separating inputs (Specifications) from outputs (Workspaces) and
execution contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AssignmentSpec(BaseModel):
    """Static definition of an assignment's inputs (Immutable).

    Attributes:
        assignment_id: Stable assignment identifier.
        job_config_path: Path to job config TOML.
        data_config_path: Path to data config TOML.
        checkpoint_path: Optional explicit checkpoint path.
    """

    assignment_id: str = Field(..., description="Stable assignment identifier")
    dataset_id: str | None = Field(default=None, description="Registry dataset identifier")
    job_id: str | None = Field(default=None, description="Registry job identifier")
    job_config_path: Path = Field(..., description="Path to job config")
    data_config_path: Path = Field(..., description="Path to data config")
    checkpoint_path: Path | None = Field(
        default=None,
        description="Explicit checkpoint path (optional)",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class AssignmentWorkspace:
    """Resolved filesystem layout for assignment artifacts.

    Attributes:
        dataset_id: Dataset identifier (for organizing assignments).
        run_id: Model/run identifier (for organizing within dataset).
        root_dir: Assignment root directory (for this specific run).
        data_dir: Directory containing input data artifacts (manifest + arrays).
    """

    dataset_id: str
    run_id: str
    root_dir: Path
    data_dir: Path

    @property
    def checkpoint_dir(self) -> Path:
        """Checkpoint directory under root."""
        return self.root_dir / "checkpoints"

    @property
    def figures_dir(self) -> Path:
        """Figures directory under root."""
        return self.root_dir / "figures"

    @property
    def predictions_dir(self) -> Path:
        """Predictions directory under root."""
        return self.root_dir / "predictions"


@dataclass(frozen=True)
class RunnableAssignment:
    """Fully resolved assignment ready for execution.

    Combines the 'what' (Spec), the 'where' (Workspace), and the
    'how' (Settings).

    Attributes:
        spec: Assignment specification.
        workspace: Filesystem workspace.
        settings: DLKit workflow settings with injected paths.
        mlflow_env: Resolved MLflow environment (tracking/artifact URIs) for
            this assignment's case, already reflected in ``settings.tracking``.
    """

    spec: AssignmentSpec
    workspace: AssignmentWorkspace
    settings: Any  # DLKit workflow settings (avoid import)
    mlflow_env: dict[str, str]


@dataclass(frozen=True)
class AssignmentBatch:
    """Complete definition of a batch workload.

    Attributes:
        output_root: Master output root for all assignments.
        assignments: List of runnable assignments.
    """

    output_root: Path
    assignments: list[RunnableAssignment]
