"""Regression tests: generation and comparison scaling stay consistent.

Guards the invariant `A @ x = b` end-to-end through the real generation and
comparison pipeline (no simulation), and that RHS is never rescaled 0/2 times.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison._linear_system import (
    ScalingMismatchError,
    prepare_resolved_linear_system,
)
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.domain.normalization import scale_factor
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.system_scaling import GershgorinScaling, NoScaling, RhsScaling
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind
from tests.dataset_support import build_test_dataset

pytestmark = pytest.mark.deferred_io("dataset-fixture-io")


def _build_normalized_dataset(root: Path) -> Path:
    """Build a `scaling=GershgorinScaling()` dataset with a spectral radius far from 1.

    A large, non-trivial scale ensures a real double-normalization bug would
    fail loudly (not be masked by a near-1.0 coincidence).
    """
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((6, 6))
    matrix = raw.T @ raw + np.eye(6) * 80.0
    matrix_path = root / "matrix.npy"
    np.save(matrix_path, matrix)
    dataset_dir = root / "dataset"
    build_test_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 3},
                seed=42,
                shuffle=False,
            ),
            scaling=GershgorinScaling(),
        ),
        str(dataset_dir),
        dataset_format="hdf5",
    )
    return dataset_dir


def test_dataset_sourced_comparison_recovers_true_solution(tmp_path: Path) -> None:
    """DATASET-sourced comparison must not double-normalize RHS."""
    dataset_dir = _build_normalized_dataset(tmp_path)
    resolved = resolve_comparison_input(
        matrix_provenance=SharedSingleMatrix(raw_matrix_index=0),
        matrix_path=dataset_dir,
        matrix_dataset_id="norm-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=1,
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        rhs_source_params={"sample_index": 0},
    )
    assert resolved.matrix_scaling is not None
    assert resolved.matrix_scaling.scaling == GershgorinScaling()
    assert resolved.lhs is not None

    system = prepare_resolved_linear_system(resolved, GershgorinScaling(), RhsScaling.NONE)
    x_exact = np.linalg.solve(system.matrix.to_dense(), system.rhs)
    np.testing.assert_allclose(x_exact, resolved.lhs, rtol=1e-8, atol=1e-8)


def test_case_scaling_different_from_the_dataset_raises(tmp_path: Path) -> None:
    """A stored matrix is never rescaled, so a case asking for another scaling must fail."""
    dataset_dir = _build_normalized_dataset(tmp_path)
    resolved = resolve_comparison_input(
        matrix_provenance=SharedSingleMatrix(raw_matrix_index=0),
        matrix_path=dataset_dir,
        matrix_dataset_id="norm-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=1,
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        rhs_source_params={"sample_index": 0},
    )

    with pytest.raises(ScalingMismatchError) as error:
        prepare_resolved_linear_system(resolved, NoScaling(), RhsScaling.NONE)

    assert (error.value.dataset_scaling, error.value.case_scaling) == (
        GershgorinScaling(),
        NoScaling(),
    )


@pytest.fixture
def two_matrix_dataset(tmp_path: Path) -> tuple[Path, tuple[np.ndarray, np.ndarray]]:
    """Two matrices, ids 3 and 7 (stored rows 0 and 1), with far apart Gershgorin scales.

    The ids differ from the stored rows, so a lookup by row instead of by id would fail.
    """
    rng = np.random.default_rng(3)
    matrices = []
    for matrix_id, shift in ((3, 10.0), (7, 1000.0)):
        raw = rng.standard_normal((6, 6))
        matrix = raw.T @ raw + np.eye(6) * shift
        np.save(tmp_path / f"A_{matrix_id}.npy", matrix)
        matrices.append(matrix)
    dataset_dir = tmp_path / "two_matrices"
    build_test_dataset(
        SourceSpec(matrix_path=str(tmp_path / "A_*.npy")),
        DatasetSpec(
            mixture=MixtureSpec(counts={"neutral_ones": 2}, seed=42, shuffle=False),
            scaling=GershgorinScaling(),
        ),
        str(dataset_dir),
        dataset_format="hdf5",
    )
    return dataset_dir, (matrices[0], matrices[1])


def test_raw_rhs_is_divided_by_the_alpha_of_its_own_stored_matrix(
    tmp_path: Path, two_matrix_dataset: tuple[Path, tuple[np.ndarray, np.ndarray]]
) -> None:
    dataset_dir, raw_matrices = two_matrix_dataset
    raw_rhs = np.arange(1.0, 7.0)
    rhs_path = tmp_path / "rhs.txt"
    np.savetxt(rhs_path, raw_rhs)
    alphas = tuple(scale_factor(matrix, GershgorinScaling()) for matrix in raw_matrices)

    scaled = []
    for matrix_row in (0, 1):
        resolved = resolve_comparison_input(
            matrix_provenance=SharedSingleMatrix(raw_matrix_index=matrix_row),
            matrix_path=dataset_dir,
            matrix_dataset_id="two-matrices",
            matrix_index=matrix_row,
            require_non_residual_rhs=True,
            seed=None,
            rhs_source_kind=ComparisonRhsSourceKind.RAW_RHS,
            rhs_source_params={"path": rhs_path, "row_kind": RowKind.STANDARD},
        )
        system = prepare_resolved_linear_system(resolved, GershgorinScaling(), RhsScaling.NONE)
        scaled.append(system.rhs.numpy())

    assert read_dataset_manifest(dataset_dir).scaling.matrix_scales.keys() == {3, 7}
    assert alphas[1] > 5 * alphas[0]
    np.testing.assert_allclose(scaled[0], raw_rhs / alphas[0], rtol=1e-14)
    np.testing.assert_allclose(scaled[1], raw_rhs / alphas[1], rtol=1e-14)


def test_gaussian_rhs_never_rescaled_by_matrix_scaling(tmp_path: Path) -> None:
    """A synthetic GAUSSIAN RHS is independently configured — never touched by matrix scale."""
    dataset_dir = _build_normalized_dataset(tmp_path)
    resolved = resolve_comparison_input(
        matrix_provenance=SharedSingleMatrix(raw_matrix_index=0),
        matrix_path=dataset_dir,
        matrix_dataset_id="norm-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=5,
        rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
        rhs_source_params={"mean": 0.0, "std": 1.0},
    )
    rhs_before = resolved.rhs.copy()

    system = prepare_resolved_linear_system(resolved, GershgorinScaling(), RhsScaling.NONE)
    np.testing.assert_array_equal(system.rhs, rhs_before)


def test_raw_lhs_source_satisfies_ax_equals_b(tmp_path: Path) -> None:
    """RAW_LHS-derived RHS stays consistent with the final matrix, raw or dataset-sourced."""
    solution_path = tmp_path / "solution.txt"
    lhs = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
    np.savetxt(solution_path, lhs)

    # Case 1: raw matrix (no manifest) — matrix and rhs scaled together.
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((6, 6))
    raw_matrix = raw.T @ raw + np.eye(6) * 40.0
    raw_matrix_path = tmp_path / "raw_matrix.npy"
    np.save(raw_matrix_path, raw_matrix)

    resolved_raw = resolve_comparison_input(
        matrix_provenance=SharedSingleMatrix(raw_matrix_index=0),
        matrix_path=raw_matrix_path,
        matrix_dataset_id="raw-matrix",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=None,
        rhs_source_kind=ComparisonRhsSourceKind.RAW_LHS,
        rhs_source_params={"path": solution_path, "row_kind": RowKind.STANDARD, "scale": 3.0},
    )
    assert resolved_raw.matrix_scaling is None
    system_raw = prepare_resolved_linear_system(resolved_raw, GershgorinScaling(), RhsScaling.NONE)
    lhs_tensor = torch.from_numpy(lhs)
    np.testing.assert_allclose(
        system_raw.matrix @ lhs_tensor, system_raw.rhs / 3.0, rtol=1e-8, atol=1e-8
    )

    # Case 2: dataset-sourced matrix (already scaled) — matrix/rhs untouched.
    dataset_dir = _build_normalized_dataset(tmp_path)
    resolved_dataset = resolve_comparison_input(
        matrix_provenance=SharedSingleMatrix(raw_matrix_index=0),
        matrix_path=dataset_dir,
        matrix_dataset_id="norm-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=None,
        rhs_source_kind=ComparisonRhsSourceKind.RAW_LHS,
        rhs_source_params={"path": solution_path, "row_kind": RowKind.STANDARD, "scale": 3.0},
    )
    assert resolved_dataset.matrix_scaling is not None
    system_dataset = prepare_resolved_linear_system(
        resolved_dataset, GershgorinScaling(), RhsScaling.NONE
    )
    np.testing.assert_allclose(
        system_dataset.matrix @ lhs_tensor, system_dataset.rhs / 3.0, rtol=1e-8, atol=1e-8
    )
