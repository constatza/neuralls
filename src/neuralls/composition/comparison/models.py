"""Composition-layer orchestration models for comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from neuralls.domain.solver.models.result import CGComparisonResult, ComparisonResult
from neuralls.platform.config.models.preconditioner_family import PreconditionerFamilyKey
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.storage.manifest import DatasetNormalization
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind

__all__ = [
    "ComparisonOutcome",
    "ComparisonParams",
    "ComparisonPaths",
    "ComparisonResult",
    "ComparisonSpec",
    "LinearSystem",
    "PreconditionerComparisonEntry",
    "ResolvedComparisonInput",
]


@dataclass(frozen=True)
class ComparisonSpec:
    """Inputs needed to run a single comparison.

    Bundles all paths and resolved artifacts required by the comparison
    orchestrator. Immutable — constructed once by the batch runner.

    Args:
        comparison_id: Stable identifier for this comparison.
        comparison_display_name: Human-readable label.
        model_config: Path to model configuration TOML.
        data_config: Path to data configuration TOML.
        comparison_config: Path to comparison configuration TOML.
        workspace: Resolved assignment workspace.
        checkpoint: Path to model checkpoint.
        matrix_override: Optional matrix file override.
        rhs_override: Optional rhs file override.
        figures_dir: Optional figures directory override.
        output_dir: Optional output directory override.
    """

    comparison_id: str
    comparison_display_name: str
    model_config: Path
    data_config: Path
    comparison_config: Path
    workspace: AssignmentWorkspace
    checkpoint: Path
    matrix_override: Path | None = None
    rhs_override: Path | None = None
    figures_dir: Path | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class ComparisonPaths:
    """Resolved paths for a single comparison run.

    Attributes:
        matrix: Path to system matrix file (.txt or .npy).
        rhs: Path to right-hand side vector file (.txt or .npy).
        output: Root output directory for comparison results.
        figures: Directory for diagnostic plots.
    """

    matrix: Path
    rhs: Path
    output: Path
    figures: Path


@dataclass(frozen=True)
class LinearSystem:
    """Loaded and validated linear system (A, b) pair.

    Attributes:
        matrix: System matrix A in Ax=b (shape: n x n).
        rhs: Right-hand side vector b in Ax=b (shape: n,).
    """

    matrix: torch.Tensor
    rhs: torch.Tensor


@dataclass(frozen=True)
class PreconditionerComparisonEntry:
    """Outcome of comparing one preconditioner: solve result, condition number, plot label.

    Attributes:
        name: Preconditioner config name.
        result: CG solver outcome for this preconditioner.
        condition_number: Effective condition number of the preconditioned system.
        label: Descriptive plot label built from the constructed preconditioner instance.
        family: Plot-style family key (see ``preconditioner_family.preconditioner_family``),
            grouping same-family preconditioners under a shared marker/linestyle/colormap.
    """

    name: str
    result: CGComparisonResult
    condition_number: float
    label: str
    family: PreconditionerFamilyKey


@dataclass(frozen=True)
class ResolvedComparisonInput:
    """Resolved comparison system and provenance used for execution and tracking."""

    matrix: np.ndarray
    rhs: np.ndarray
    matrix_dataset_id: str
    matrix_index: int
    lhs: np.ndarray | None = None
    rhs_dataset_id: str | None = None
    rhs_sample_index: int | None = None
    rhs_kind: RowKind | None = None
    rhs_source_kind: ComparisonRhsSourceKind | None = None
    rhs_source_params: dict[str, object] | None = None
    matrix_normalization: DatasetNormalization | None = None
    """Persisted normalization metadata for the dataset the matrix was loaded
    from, or None when the matrix came from a raw/external file with no
    manifest. Drives whether comparison-time code treats the matrix as
    already-normalized (skip/derive) or genuinely raw (compute fresh)."""

    @property
    def rhs_source_type(self) -> str:
        match self.rhs_source_kind:
            case ComparisonRhsSourceKind.GAUSSIAN | ComparisonRhsSourceKind.SPARSE:
                return "generated"
            case None:
                return "unknown"
            case _:
                return str(self.rhs_source_kind)


@dataclass(frozen=True)
class ComparisonParams:
    """Runtime parameters for comparison execution.

    Attributes:
        force: Rerun every comparison even if a matching one (same
            comparison_id and resolved preconditioner checkpoints) already
            completed successfully.
    """

    force: bool = False


@dataclass(frozen=True)
class ComparisonOutcome:
    """Result for a single comparison run.

    Args:
        comparison_id: Stable identifier for this comparison.
        comparison_display_name: Human-readable label.
        success: Whether the comparison completed without error.
        error: Error message if failed.
        payload: Full comparison result if successful.
        warnings: Non-fatal warning messages.
    """

    comparison_id: str
    comparison_display_name: str
    success: bool
    error: str | None = None
    payload: ComparisonResult | None = None
    warnings: tuple[str, ...] = ()
