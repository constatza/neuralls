from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.source_streams import EnumerateBy
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.shared.enum_codecs import encode_row_kind_array
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind


def _build_safe_dataset(root: Path, *, residual: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
    matrix_path = root / "matrix.npy"
    np.save(matrix_path, matrix)
    dataset_dir = root / ("residual_dataset" if residual else "safe_dataset")
    build_dataset(
        SourceSpec(
            matrix_path=str(matrix_path),
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"neutral_ones": 2 if residual else 3},
                seed=7 if residual else 42,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(dataset_dir),
        dataset_format="npy",
    )
    return dataset_dir


def test_resolve_comparison_input_selects_canonical_dataset_triplet(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path)

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="safe-dataset",
        matrix_index=None,
        require_non_residual_rhs=True,
        seed=11,
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        rhs_source_params={"path": dataset_dir},
    )

    assert resolved.rhs_source_type == "dataset"
    assert resolved.rhs_dataset_id == str(dataset_dir)
    assert resolved.rhs_sample_index == 0
    assert resolved.rhs_kind is RowKind.STANDARD
    assert resolved.lhs is not None
    assert resolved.matrix.shape == (2, 2)
    assert resolved.rhs.shape == (2,)


def test_resolve_comparison_input_rejects_unsafe_dataset_triplet(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path, residual=True)
    np.save(
        dataset_dir / "row_kind.npy",
        encode_row_kind_array([RowKind.CG_INTERNAL, RowKind.CG_INTERNAL]),
    )

    with pytest.raises(ValueError, match="No canonical non-residual rows"):
        resolve_comparison_input(
            matrix_path=dataset_dir,
            matrix_dataset_id="residual-dataset",
            matrix_index=None,
            require_non_residual_rhs=True,
            seed=3,
            rhs_source_kind=ComparisonRhsSourceKind.DATASET,
            rhs_source_params={"path": dataset_dir},
        )


def test_resolve_comparison_input_generates_gaussian_rhs(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path)

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="safe-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=5,
        rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
        rhs_source_params={"mean": 0.0, "std": 1.0},
    )

    assert resolved.rhs_source_type == "generated"
    assert resolved.rhs_source_kind is ComparisonRhsSourceKind.GAUSSIAN
    assert resolved.rhs.shape == (2,)
    assert resolved.rhs_dataset_id is None
    assert resolved.lhs is None


def test_resolve_comparison_input_generates_sparse_rhs(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path)

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="safe-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=None,
        rhs_source_kind=ComparisonRhsSourceKind.SPARSE,
        rhs_source_params={"indices": [0, 1], "values": [3.0, -2.0]},
    )

    np.testing.assert_allclose(resolved.rhs, np.array([3.0, -2.0], dtype=np.float64))
    assert resolved.rhs_source_kind is ComparisonRhsSourceKind.SPARSE


def test_resolve_comparison_input_loads_raw_rhs_with_explicit_row_kind(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path)
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, np.array([[9.0, 8.0], [3.0, -2.0]], dtype=np.float64))

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="safe-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=None,
        rhs_source_kind=ComparisonRhsSourceKind.RAW_RHS,
        rhs_source_params={
            "path": rhs_path,
            "sample_index": 1,
            "row_kind": RowKind.STANDARD,
        },
    )

    np.testing.assert_allclose(resolved.rhs, np.array([3.0, -2.0], dtype=np.float64))
    assert resolved.rhs_source_type == "raw_rhs"
    assert resolved.rhs_sample_index == 1
    assert resolved.rhs_kind is RowKind.STANDARD
    assert resolved.rhs_source_kind is ComparisonRhsSourceKind.RAW_RHS
    assert resolved.lhs is None


def test_resolve_comparison_input_scales_raw_lhs_into_rhs(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path)
    solution_path = tmp_path / "solution.txt"
    np.savetxt(solution_path, np.array([2.0, -1.0], dtype=np.float64))

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="safe-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=None,
        rhs_source_kind=ComparisonRhsSourceKind.RAW_LHS,
        rhs_source_params={
            "path": solution_path,
            "row_kind": RowKind.STANDARD,
            "scale": 5.0,
        },
    )

    np.testing.assert_allclose(resolved.rhs, np.array([45.0, -25.0], dtype=np.float64))
    assert resolved.lhs is not None
    np.testing.assert_allclose(resolved.lhs, np.array([2.0, -1.0], dtype=np.float64))
    assert resolved.rhs_source_type == "raw_lhs"
    assert resolved.rhs_sample_index == 0
    assert resolved.rhs_kind is RowKind.STANDARD
    assert resolved.rhs_source_kind is ComparisonRhsSourceKind.RAW_LHS


def test_resolve_comparison_input_rejects_raw_rhs_without_row_kind(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path)
    rhs_path = tmp_path / "rhs.txt"
    np.savetxt(rhs_path, np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ValueError, match="must define row_kind"):
        resolve_comparison_input(
            matrix_path=dataset_dir,
            matrix_dataset_id="safe-dataset",
            matrix_index=0,
            require_non_residual_rhs=True,
            seed=None,
            rhs_source_kind=ComparisonRhsSourceKind.RAW_RHS,
            rhs_source_params={"path": rhs_path},
        )


def _write_distinct_spd_matrices(mat_dir: Path, count: int, n: int = 3) -> list[np.ndarray]:
    mat_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    matrices = []
    for i in range(count):
        A = rng.standard_normal((n, n))
        A = A @ A.T + n * np.eye(n)
        np.savetxt(mat_dir / f"matrix_{i:03d}.txt", A)
        matrices.append(A)
    return matrices


def test_resolve_comparison_input_matrix_index_selects_distinct_matrix_with_one_row_per_matrix(
    tmp_path: Path,
) -> None:
    """A dataset built with exactly one sample per matrix must expose distinct matrix_index values.

    Regression guard: a comparison/holdout dataset for a parametric matrix family only
    identifies genuinely distinct family members via `matrix_index` when its generation
    strategy's sample count equals the number of included matrices (deterministic
    one-row-per-binding allocation, see `_allocate_strategy_counts_across_bindings`).
    """
    mat_dir = tmp_path / "matrices"
    matrices = _write_distinct_spd_matrices(mat_dir, count=3)
    dataset_dir = tmp_path / "holdout_dataset"
    build_dataset(
        SourceSpec(
            matrix_path=str(mat_dir / "matrix_*.txt"),
            enumerate_by=EnumerateBy.NAME,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": 3},
                seed=0,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(dataset_dir),
        dataset_format="npy",
    )

    for index, expected in enumerate(matrices):
        resolved = resolve_comparison_input(
            matrix_path=dataset_dir,
            matrix_dataset_id="holdout-dataset",
            matrix_index=index,
            require_non_residual_rhs=True,
            seed=1,
            rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
            rhs_source_params={"mean": 0.0, "std": 1.0},
        )
        np.testing.assert_allclose(resolved.matrix, expected)


def test_resolve_comparison_input_matrix_index_can_collide_when_samples_are_pooled(
    tmp_path: Path,
) -> None:
    """Pinning test for a real footgun: when a strategy's sample budget exceeds the number of
    matrices, rows are grouped into per-matrix blocks (no cross-binding shuffle), so small
    `matrix_index` values can all resolve to the SAME physical matrix instead of distinct
    family members. A comparison/holdout dataset must set `samples == num_matrices` to avoid
    this — see `configs/datasets/.../holdout-matrices.toml`.
    """
    mat_dir = tmp_path / "matrices"
    matrices = _write_distinct_spd_matrices(mat_dir, count=3)
    dataset_dir = tmp_path / "pooled_dataset"
    build_dataset(
        SourceSpec(
            matrix_path=str(mat_dir / "matrix_*.txt"),
            enumerate_by=EnumerateBy.NAME,
        ),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": 9},
                seed=0,
                shuffle=False,
            ),
            normalize="none",
        ),
        str(dataset_dir),
        dataset_format="npy",
    )

    resolved_0 = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="pooled-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=1,
        rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
        rhs_source_params={"mean": 0.0, "std": 1.0},
    )
    resolved_1 = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="pooled-dataset",
        matrix_index=1,
        require_non_residual_rhs=True,
        seed=1,
        rhs_source_kind=ComparisonRhsSourceKind.GAUSSIAN,
        rhs_source_params={"mean": 0.0, "std": 1.0},
    )
    # Both positions fall inside the first matrix's 3-row block: same physical matrix.
    np.testing.assert_allclose(resolved_0.matrix, resolved_1.matrix)
    np.testing.assert_allclose(resolved_0.matrix, matrices[0])


def test_resolve_comparison_input_loads_explicit_dataset_triplet(tmp_path: Path) -> None:
    matrix_dataset = _build_safe_dataset(tmp_path / "matrix")
    rhs_source = _build_safe_dataset(tmp_path / "rhs")

    resolved = resolve_comparison_input(
        matrix_path=matrix_dataset,
        matrix_dataset_id="matrix-dataset",
        matrix_index=0,
        require_non_residual_rhs=True,
        seed=None,
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        rhs_source_params={"path": rhs_source, "sample_index": 1},
    )

    assert resolved.rhs_source_type == "dataset"
    assert resolved.rhs_dataset_id == str(rhs_source)
    assert resolved.rhs_sample_index == 1
    assert resolved.rhs_kind is RowKind.STANDARD
    assert resolved.rhs_source_kind is ComparisonRhsSourceKind.DATASET
    assert resolved.lhs is not None
    assert resolved.matrix.shape == (2, 2)
    assert resolved.rhs.shape == (2,)
