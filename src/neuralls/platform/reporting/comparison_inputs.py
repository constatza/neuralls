"""Artifact staging for resolved comparison input systems."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuralls.domain.solver.models.provenance import (
    CrossDatasetMatrix,
    MatrixProvenance,
    PinnedSplitMatrix,
    UnverifiedMatrix,
)
from neuralls.platform.reporting.serialization import to_json_primitive
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind


def _matrix_provenance_to_json(provenance: MatrixProvenance | None) -> dict[str, object] | None:
    """Serialize one `MatrixProvenance` variant into a tagged JSON-safe dict."""
    match provenance:
        case None:
            return None
        case PinnedSplitMatrix(test_position=test_position, raw_matrix_index=raw_matrix_index):
            return {
                "kind": "pinned_split",
                "test_position": int(test_position),
                "raw_matrix_index": int(raw_matrix_index),
            }
        case UnverifiedMatrix(raw_matrix_index=raw_matrix_index):
            return {
                "kind": "unverified",
                "raw_matrix_index": int(raw_matrix_index),
            }
        case CrossDatasetMatrix(dataset_id=dataset_id, raw_matrix_index=raw_matrix_index):
            return {
                "kind": "cross_dataset",
                "dataset_id": dataset_id,
                "raw_matrix_index": int(raw_matrix_index),
            }


@dataclass(frozen=True)
class ComparisonInputArtifacts:
    """Platform-owned payload for staging resolved comparison inputs."""

    matrix: np.ndarray
    rhs: np.ndarray
    matrix_dataset_id: str
    matrix_index: int
    rhs_source_type: str
    lhs: np.ndarray | None = None
    rhs_dataset_id: str | None = None
    rhs_sample_index: int | None = None
    row_kind: RowKind | None = None
    rhs_source_kind: ComparisonRhsSourceKind | None = None
    rhs_source_params: dict[str, object] | None = None
    matrix_provenance: MatrixProvenance | None = None


def stage_comparison_inputs(root: Path, resolved: ComparisonInputArtifacts) -> Path:
    """Persist the resolved `(A, b)` pair and provenance for MLflow upload."""
    output_dir = root / "comparison_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "matrix.npy", resolved.matrix)
    np.save(output_dir / "rhs.npy", resolved.rhs)
    if resolved.lhs is not None:
        np.save(output_dir / "lhs.npy", resolved.lhs)
    provenance = {
        "matrix_dataset_id": resolved.matrix_dataset_id,
        "matrix_index": resolved.matrix_index,
        "rhs_source_type": resolved.rhs_source_type,
        "rhs_dataset_id": resolved.rhs_dataset_id,
        "rhs_sample_index": resolved.rhs_sample_index,
        "row_kind": str(resolved.row_kind) if resolved.row_kind is not None else None,
        "rhs_source_kind": (
            str(resolved.rhs_source_kind) if resolved.rhs_source_kind is not None else None
        ),
        "rhs_source_params": to_json_primitive(resolved.rhs_source_params),
        "matrix_provenance": _matrix_provenance_to_json(resolved.matrix_provenance),
        "lhs_available": resolved.lhs is not None,
        "matrix_shape": list(resolved.matrix.shape),
        "rhs_shape": list(resolved.rhs.shape),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir
