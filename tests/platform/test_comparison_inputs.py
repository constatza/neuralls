from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neuralls.domain.solver.models.provenance import PinnedSplitMatrix, UnverifiedMatrix
from neuralls.platform.reporting.comparison_inputs import (
    ComparisonInputArtifacts,
    stage_comparison_inputs,
)
from neuralls.shared.types import ComparisonRhsSourceKind, MatrixId, RowKind, TestPosition


def test_stage_comparison_inputs_writes_artifacts(tmp_path: Path) -> None:
    resolved = ComparisonInputArtifacts(
        matrix=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        rhs=np.array([1.0, 4.0], dtype=np.float64),
        matrix_dataset_id="matrix-ds",
        matrix_index=0,
        rhs_source_type="generated",
        row_kind=RowKind.STANDARD,
        rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
        rhs_source_params={"mean": 0.0, "std": 1.0},
    )

    output_dir = stage_comparison_inputs(tmp_path, resolved)

    assert (output_dir / "matrix.npy").exists()
    assert (output_dir / "rhs.npy").exists()
    assert (output_dir / "provenance.json").exists()
    np.testing.assert_allclose(np.load(output_dir / "matrix.npy"), resolved.matrix)
    np.testing.assert_allclose(np.load(output_dir / "rhs.npy"), resolved.rhs)
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["matrix_dataset_id"] == "matrix-ds"
    assert provenance["rhs_source_kind"] == "gaussian"
    assert provenance["lhs_available"] is False
    assert provenance["row_kind"] == "standard"
    assert "rhs_kind" not in provenance


def test_stage_comparison_inputs_serializes_pinned_split_provenance(tmp_path: Path) -> None:
    resolved = ComparisonInputArtifacts(
        matrix=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        rhs=np.array([1.0, 4.0], dtype=np.float64),
        matrix_dataset_id="matrix-ds",
        matrix_index=2,
        rhs_source_type="dataset",
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        matrix_provenance=PinnedSplitMatrix(
            test_position=TestPosition(0), raw_matrix_index=MatrixId(2)
        ),
    )

    output_dir = stage_comparison_inputs(tmp_path, resolved)
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))

    assert provenance["matrix_provenance"] == {
        "kind": "pinned_split",
        "test_position": 0,
        "raw_matrix_index": 2,
    }


def test_stage_comparison_inputs_serializes_unverified_provenance(tmp_path: Path) -> None:
    resolved = ComparisonInputArtifacts(
        matrix=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        rhs=np.array([1.0, 4.0], dtype=np.float64),
        matrix_dataset_id="matrix-ds",
        matrix_index=0,
        rhs_source_type="generated",
        matrix_provenance=UnverifiedMatrix(raw_matrix_index=MatrixId(0)),
    )

    output_dir = stage_comparison_inputs(tmp_path, resolved)
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))

    assert provenance["matrix_provenance"] == {"kind": "unverified", "raw_matrix_index": 0}
