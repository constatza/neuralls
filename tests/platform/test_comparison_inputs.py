from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.solver.models.provenance import (
    HeldOutMatrix,
    MatrixProvenance,
    SharedSingleMatrix,
)
from neuralls.platform.reporting.comparison_inputs import (
    ComparisonInputArtifacts,
    stage_comparison_inputs,
)
from neuralls.shared.types import ComparisonRhsSourceKind, MatrixId, RowKind


def test_stage_comparison_inputs_writes_artifacts(tmp_path: Path) -> None:
    resolved = ComparisonInputArtifacts(
        matrix=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        rhs=np.array([1.0, 4.0], dtype=np.float64),
        matrix_dataset_id="matrix-ds",
        matrix_index=0,
        rhs_source_type="generated",
        matrix_provenance=SharedSingleMatrix(raw_matrix_index=0),
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


def _artifacts(provenance: MatrixProvenance, *, matrix_index: int) -> ComparisonInputArtifacts:
    return ComparisonInputArtifacts(
        matrix=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        rhs=np.array([1.0, 4.0], dtype=np.float64),
        matrix_dataset_id="matrix-ds",
        matrix_index=matrix_index,
        rhs_source_type="generated",
        matrix_provenance=provenance,
    )


@pytest.mark.parametrize(
    ("provenance", "expected"),
    [
        (
            SharedSingleMatrix(raw_matrix_index=0),
            {"kind": "shared_single", "raw_matrix_index": 0},
        ),
        (
            HeldOutMatrix(matrix_id=MatrixId(9), raw_matrix_index=2),
            {"kind": "held_out", "matrix_id": 9, "raw_matrix_index": 2},
        ),
    ],
)
def test_stage_comparison_inputs_serializes_each_provenance_variant(
    tmp_path: Path, provenance: MatrixProvenance, expected: dict[str, object]
) -> None:
    output_dir = stage_comparison_inputs(tmp_path, _artifacts(provenance, matrix_index=2))

    payload = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))

    assert payload["matrix_provenance"] == expected
