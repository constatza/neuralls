"""Composition-layer orchestration models for comparison workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from neuralls.domain.solver.models.provenance import MatrixProvenance
from neuralls.domain.solver.models.result import CGComparisonResult, ComparisonResult
from neuralls.platform.config.models.preconditioner_family import PreconditionerFamilyKey
from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.storage.manifest import DatasetNormalization
from neuralls.shared.types import ComparisonRhsSourceKind, DatasetId, MatrixId, RowKind

__all__ = [
    "ComparisonGroup",
    "ComparisonGroupings",
    "ComparisonOutcome",
    "ComparisonParams",
    "ComparisonPaths",
    "ComparisonResult",
    "ComparisonSpec",
    "GroupMatrixIdentity",
    "LinearSystem",
    "MultiMatrixDataset",
    "PreconditionerComparisonEntry",
    "ResolvedComparisonInput",
    "SingleMatrixDataset",
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
    """Outcome of comparing one preconditioner: solve result and plot metadata.

    Attributes:
        name: Preconditioner config name.
        result: CG solver outcome for this preconditioner.
        label: Descriptive plot label built from the constructed preconditioner instance.
        family: Plot-style family key (see ``preconditioner_family.preconditioner_family``),
            driving shared linestyle across same-family preconditioners.
        color_key: Optional plot color-axis key (e.g. a POD-2G fit dataset).
            Entries sharing a key share a color; ``None`` falls back to ``family``.
        marker_key: Optional plot marker-axis key (e.g. a POD-2G weighting
            scheme). Entries sharing a key share a marker; ``None`` falls back
            to ``family``.
    """

    name: str
    result: CGComparisonResult
    label: str
    family: PreconditionerFamilyKey
    color_key: str | None = None
    marker_key: str | None = None


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
    row_kind: RowKind | None = None
    rhs_source_kind: ComparisonRhsSourceKind | None = None
    rhs_source_params: dict[str, object] | None = None
    matrix_normalization: DatasetNormalization | None = None
    """Persisted normalization metadata for the dataset the matrix was loaded
    from, or None when the matrix came from a raw/external file with no
    manifest. Drives whether comparison-time code treats the matrix as
    already-normalized (skip/derive) or genuinely raw (compute fresh)."""
    matrix_provenance: MatrixProvenance | None = None
    """Whether this matrix was verified held-out via a pinned split, left
    unverified (a raw matrix_index), or loaded cross-dataset. `None` only
    for call sites that have not yet been migrated to attach it; new result
    producers should always set it."""

    @property
    def rhs_source_type(self) -> str:
        match self.rhs_source_kind:
            case (
                ComparisonRhsSourceKind.GAUSSIAN
                | ComparisonRhsSourceKind.SPARSE
                | ComparisonRhsSourceKind.SINUSOIDAL
            ):
                return "generated"
            case None:
                return "unknown"
            case _:
                return str(self.rhs_source_kind)


@dataclass(frozen=True)
class ComparisonParams:
    """Runtime parameters for comparison execution.

    Attributes:
        force: Rerun every comparison even if a matching one (same derived
            identity: config, data, preconditioners and checkpoint content)
            already completed successfully.
    """

    force: bool = False


@dataclass(frozen=True, slots=True)
class SingleMatrixDataset:
    """This dataset has exactly one matrix — no sample index exists at all."""

    dataset_id: DatasetId


@dataclass(frozen=True, slots=True)
class MultiMatrixDataset:
    """This dataset stores several matrix samples.

    `resolved_indices` are the ones a comparison group's members actually
    resolved to.
    """

    dataset_id: DatasetId
    resolved_indices: frozenset[MatrixId]


type GroupMatrixIdentity = SingleMatrixDataset | MultiMatrixDataset
"""Which matrix(es) a `ComparisonGroup`'s members resolve to.

Exactly two variants because a dataset is exactly one of two kinds (see
`platform.storage.dataset_readers.list_available_matrix_indices`). For a
`MultiMatrixDataset`, `len(resolved_indices) == 1` means every member hit
the same matrix — a derived fact, not a third variant.
"""


@dataclass(frozen=True, slots=True)
class ComparisonGroup:
    """Every resolved comparison entry sharing one partition key.

    Hand-written `[[comparisons]]` entries and `comparison_sweeps`-expanded
    entries land in the same group whenever their data says they belong
    together under whichever key partitioned them — group membership is a
    property of what an entry tests, never of which TOML array declared
    it. See `ComparisonGroupings` for the two partition keys this codebase
    actually uses.
    """

    group_id: str
    matrix_identity: GroupMatrixIdentity
    entry_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class ComparisonGroupings:
    """Every comparison group discovered for one batch, under both granularities.

    `by_matrix` partitions purely by `matrix_dataset` — this is the
    execution-time grouping `SweepPreconditionerCache` scopes
    setup-sharing to, since `rhs_source.kind` has no bearing on whether a
    preconditioner's `setup()` can be reused (`setup()` never sees the RHS
    at all). `by_matrix_and_rhs_kind` additionally partitions by
    `rhs_source.kind` — a reporting-only view that keeps same-matrix,
    different-RHS-kind systems apart, so a preconditioner's behavior under
    one forcing structure doesn't get blended into another's average. Both
    are reported (see `comparison_batch.py::_log_group_aggregation`), but
    only `by_matrix` ever backs a `SweepPreconditionerCache`.
    """

    by_matrix: Mapping[str, ComparisonGroup]
    by_matrix_and_rhs_kind: Mapping[str, ComparisonGroup]


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
