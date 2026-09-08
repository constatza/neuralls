"""Regression tests: generation and comparison normalization stay consistent.

Guards the invariant `A @ x = b` end-to-end through the real generation and
comparison pipeline (no simulation), and that RHS is never rescaled 0/2 times.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison._linear_system import _load_linear_system
from neuralls.composition.comparison.models import ComparisonPaths
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind


def _build_normalized_dataset(root: Path) -> Path:
    """Build a `normalize="matrix"` dataset with a spectral radius far from 1.

    A large, non-trivial scale ensures a real double-normalization bug would
    fail loudly (not be masked by a near-1.0 coincidence).
    """
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((6, 6))
    matrix = raw.T @ raw + np.eye(6) * 80.0
    matrix_path = root / "matrix.npy"
    np.save(matrix_path, matrix)
    dataset_dir = root / "dataset"
    build_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 3},
                seed=42,
                shuffle=False,
            ),
            normalize="matrix",
        ),
        str(dataset_dir),
        dataset_format="npy",
    )
    return dataset_dir


def _dummy_paths(root: Path) -> ComparisonPaths:
    return ComparisonPaths(matrix=root, rhs=root, output=root / "out", figures=root / "figs")


def test_dataset_sourced_comparison_recovers_true_solution(tmp_path: Path) -> None:
    """DATASET-sourced comparison must not double-normalize RHS."""
    dataset_dir = _build_normalized_dataset(tmp_path)
    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="norm-dataset",
        matrix_index=None,
        require_non_residual_rhs=True,
        seed=1,
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        rhs_source_params={"path": dataset_dir},
    )
    assert resolved.matrix_normalization is not None
    assert resolved.matrix_normalization.type == "matrix"
    assert resolved.lhs is not None

    system = _load_linear_system(
        _dummy_paths(tmp_path),
        rhs_sample_index=0,
        matrix_index=0,
        normalize_system="matrix",
        resolved_input=resolved,
    )
    x_exact = np.linalg.solve(system.matrix, system.rhs)
    np.testing.assert_allclose(x_exact, resolved.lhs, rtol=1e-8, atol=1e-8)


def test_dataset_sourced_comparison_matches_no_renormalization(tmp_path: Path) -> None:
    """A matching normalize_system must be an exact no-op vs 'none'."""
    dataset_dir = _build_normalized_dataset(tmp_path)
    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="norm-dataset",
        matrix_index=None,
        require_non_residual_rhs=True,
        seed=1,
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        rhs_source_params={"path": dataset_dir},
    )

    system_matrix = _load_linear_system(
        _dummy_paths(tmp_path),
        rhs_sample_index=0,
        matrix_index=0,
        normalize_system="matrix",
        resolved_input=resolved,
    )
    system_none = _load_linear_system(
        _dummy_paths(tmp_path),
        rhs_sample_index=0,
        matrix_index=0,
        normalize_system="none",
        resolved_input=resolved,
    )
    np.testing.assert_array_equal(system_matrix.matrix, system_none.matrix)
    np.testing.assert_array_equal(system_matrix.rhs, system_none.rhs)


def test_normalize_system_mismatch_raises(tmp_path: Path) -> None:
    """A stale dataset with an unsupported persisted normalization type must raise."""
    dataset_dir = _build_normalized_dataset(tmp_path)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["normalization"]["type"] = "diagonal"
    manifest_path.write_text(json.dumps(manifest))

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="norm-dataset",
        matrix_index=None,
        require_non_residual_rhs=True,
        seed=1,
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        rhs_source_params={"path": dataset_dir},
    )
    assert resolved.matrix_normalization is not None
    assert resolved.matrix_normalization.type == "diagonal"

    with pytest.raises(ValueError, match="no longer supported"):
        _load_linear_system(
            _dummy_paths(tmp_path),
            rhs_sample_index=0,
            matrix_index=0,
            normalize_system="matrix",
            resolved_input=resolved,
        )


def test_gaussian_rhs_never_rescaled_by_matrix_normalization(tmp_path: Path) -> None:
    """A synthetic GAUSSIAN RHS is independently configured — never touched by matrix scale."""
    dataset_dir = _build_normalized_dataset(tmp_path)
    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="norm-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=5,
        rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
        rhs_source_params={"mean": 0.0, "std": 1.0},
    )
    assert resolved.matrix_normalization is not None
    rhs_before = resolved.rhs.copy()

    system = _load_linear_system(
        _dummy_paths(tmp_path),
        rhs_sample_index=0,
        matrix_index=0,
        normalize_system="matrix",
        resolved_input=resolved,
    )
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
        matrix_path=raw_matrix_path,
        matrix_dataset_id="raw-matrix",
        matrix_index=None,
        require_non_residual_rhs=True,
        seed=None,
        rhs_source_kind=ComparisonRhsSourceKind.RAW_LHS,
        rhs_source_params={"path": solution_path, "row_kind": RowKind.STANDARD, "scale": 3.0},
    )
    assert resolved_raw.matrix_normalization is None
    system_raw = _load_linear_system(
        _dummy_paths(tmp_path),
        rhs_sample_index=0,
        matrix_index=0,
        normalize_system="matrix",
        resolved_input=resolved_raw,
    )
    np.testing.assert_allclose(system_raw.matrix @ lhs, system_raw.rhs / 3.0, rtol=1e-8, atol=1e-8)

    # Case 2: dataset-sourced matrix (already normalized) — matrix/rhs untouched.
    dataset_dir = _build_normalized_dataset(tmp_path)
    resolved_dataset = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="norm-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=None,
        rhs_source_kind=ComparisonRhsSourceKind.RAW_LHS,
        rhs_source_params={"path": solution_path, "row_kind": RowKind.STANDARD, "scale": 3.0},
    )
    assert resolved_dataset.matrix_normalization is not None
    system_dataset = _load_linear_system(
        _dummy_paths(tmp_path),
        rhs_sample_index=0,
        matrix_index=0,
        normalize_system="matrix",
        resolved_input=resolved_dataset,
    )
    np.testing.assert_allclose(
        system_dataset.matrix @ lhs, system_dataset.rhs / 3.0, rtol=1e-8, atol=1e-8
    )
