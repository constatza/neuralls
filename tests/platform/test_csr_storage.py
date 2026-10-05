"""Tests for CSR matrix storage: layout, writer, accumulator, and readers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_array

from neuralls.platform.storage.csr_layout import build_csr, choose_layout, split_csr
from neuralls.platform.storage.csr_storage import CsrAccumulator, write_csr_matrix_group
from neuralls.platform.storage.dataset_readers import (
    load_matrix_dense_sample,
    load_matrix_sparse_sample,
)
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import (
    load_dataset_manifest,
    make_dataset_manifest,
    save_dataset_manifest,
)
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.types import DatasetFormat, LayoutType

SEED = 0
ROWS = 7
COLS = 5
SAMPLE_COUNT = 3
DENSITY = 0.4
MATRIX_DATASET_MEMBER = "matrix"
RHS_MEMBER = "rhs"
SOLUTIONS_MEMBER = "solutions"


def _same_matrix(left: csr_array, right: csr_array) -> bool:
    """Compare two sparse matrices by their dense values, shape included."""
    return left.shape == right.shape and bool(np.array_equal(left.toarray(), right.toarray()))


def _seeded_sparse(rows: int, cols: int, seed: int) -> csr_array:
    """Return a seeded random CSR matrix with a non-trivial pattern."""
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((rows, cols)) * (rng.random((rows, cols)) < DENSITY)
    return csr_array(dense)


@pytest.fixture
def distinct_patterns() -> tuple[csr_array, ...]:
    """Samples with different sparsity patterns (per-sample layout)."""
    return tuple(_seeded_sparse(ROWS, COLS, SEED + i) for i in range(SAMPLE_COUNT))


@pytest.fixture
def shared_pattern() -> tuple[csr_array, ...]:
    """Samples that share one pattern and differ only in values (broadcast layout)."""
    base = _seeded_sparse(ROWS, COLS, SEED)
    rng = np.random.default_rng(SEED + 1)
    return tuple(
        csr_array(
            (rng.random(base.nnz), base.indices.copy(), base.indptr.copy()),
            shape=base.shape,
        )
        for _ in range(SAMPLE_COUNT)
    )


@pytest.fixture
def rhs_and_solutions() -> tuple[np.ndarray, np.ndarray]:
    """Seeded dense RHS and solution rows matching ``SAMPLE_COUNT`` samples."""
    rng = np.random.default_rng(SEED)
    return rng.standard_normal((SAMPLE_COUNT, COLS)), rng.standard_normal((SAMPLE_COUNT, COLS))


def _write_zarr_array(path: Path, data: np.ndarray) -> None:
    arr = zarr.open_array(str(path), mode="w", shape=data.shape, dtype="float64")
    arr[:] = data


def _save_manifest(
    dataset_dir: Path,
    matrix: DatasetArtifact,
    rhs: np.ndarray,
    solutions: np.ndarray,
) -> None:
    """Write rhs/solutions arrays and a manifest that points at ``matrix``."""
    _write_zarr_array(dataset_dir / RHS_MEMBER, rhs)
    _write_zarr_array(dataset_dir / SOLUTIONS_MEMBER, solutions)
    save_dataset_manifest(
        dataset_dir,
        make_dataset_manifest(
            matrix=matrix,
            rhs=DatasetArtifact(path=RHS_MEMBER, format="zarr", dtype="float64", shape=rhs.shape),
            solutions=DatasetArtifact(
                path=SOLUTIONS_MEMBER, format="zarr", dtype="float64", shape=solutions.shape
            ),
            normalization=DatasetNormalization(
                type="none", matrix_norm=1.0, matrix_norm_type="frobenius", scale={}
            ),
        ),
    )


def _csr_dataset(
    dataset_dir: Path,
    matrices: tuple[csr_array, ...],
    rhs_and_solutions: tuple[np.ndarray, np.ndarray],
) -> Path:
    """Write a complete CSR dataset (matrix group, rhs, solutions, manifest)."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    artifact = write_csr_matrix_group(
        "zarr",
        dataset_dir / MATRIX_DATASET_MEMBER,
        matrices,
        member_path=MATRIX_DATASET_MEMBER,
    )
    _save_manifest(dataset_dir, artifact, *rhs_and_solutions)
    return dataset_dir


def test_csr_layout_round_trip_preserves_values_and_structure(
    distinct_patterns: tuple[csr_array, ...],
) -> None:
    original = distinct_patterns[0]
    parts = split_csr(original)
    rebuilt = build_csr(parts.indptr, parts.indices, parts.data, parts.shape)
    assert rebuilt.shape == original.shape
    assert _same_matrix(rebuilt, original)
    assert np.array_equal(rebuilt.indptr, original.indptr)
    assert np.array_equal(rebuilt.indices, original.indices)


def test_build_csr_rejects_corrupt_indptr(distinct_patterns: tuple[csr_array, ...]) -> None:
    parts = split_csr(distinct_patterns[0])
    corrupt = parts.indptr.copy()
    corrupt[-1] += 1
    with pytest.raises(ValueError, match="indptr"):
        build_csr(corrupt, parts.indices, parts.data, parts.shape)


def test_choose_layout_broadcast_when_patterns_match(shared_pattern: tuple[csr_array, ...]) -> None:
    assert choose_layout(shared_pattern) is LayoutType.SHARED_PATTERN


def test_choose_layout_per_sample_when_patterns_differ(
    distinct_patterns: tuple[csr_array, ...],
) -> None:
    assert choose_layout(distinct_patterns) is LayoutType.MANY_MATRICES


def test_per_sample_round_trip_reads_each_matrix(
    tmp_path: Path,
    distinct_patterns: tuple[csr_array, ...],
    rhs_and_solutions: tuple[np.ndarray, np.ndarray],
) -> None:
    dataset_dir = _csr_dataset(tmp_path / "ds", distinct_patterns, rhs_and_solutions)
    for index, original in enumerate(distinct_patterns):
        loaded = load_matrix_sparse_sample(dataset_dir, index)
        assert loaded.shape == original.shape
        assert _same_matrix(loaded, original)


def test_broadcast_round_trip_reads_each_sample_values(
    tmp_path: Path,
    shared_pattern: tuple[csr_array, ...],
    rhs_and_solutions: tuple[np.ndarray, np.ndarray],
) -> None:
    dataset_dir = _csr_dataset(tmp_path / "ds", shared_pattern, rhs_and_solutions)
    manifest_matrix = load_dataset_manifest(dataset_dir)["matrix"]
    assert manifest_matrix["layout"] == LayoutType.SHARED_PATTERN.value
    for index, original in enumerate(shared_pattern):
        loaded = load_matrix_sparse_sample(dataset_dir, index)
        assert loaded.shape == original.shape
        assert _same_matrix(loaded, original)


def test_dense_reader_densifies_csr_dataset(
    tmp_path: Path,
    distinct_patterns: tuple[csr_array, ...],
    rhs_and_solutions: tuple[np.ndarray, np.ndarray],
) -> None:
    dataset_dir = _csr_dataset(tmp_path / "ds", distinct_patterns, rhs_and_solutions)
    dense = load_matrix_dense_sample(dataset_dir, 1)
    assert isinstance(dense, np.ndarray)
    np.testing.assert_array_equal(dense, distinct_patterns[1].toarray())


def test_mixed_formats_rejected(tmp_path: Path, distinct_patterns: tuple[csr_array, ...]) -> None:
    mixed = (distinct_patterns[0], distinct_patterns[1].toarray())
    with pytest.raises(ValueError, match="csr_array"):
        write_csr_matrix_group("zarr", tmp_path / "g", mixed, member_path="matrix")


@pytest.mark.parametrize("dataset_format", ["npy", "hdf5"])
def test_csr_rejected_for_non_zarr_formats(
    tmp_path: Path, dataset_format: DatasetFormat, distinct_patterns: tuple[csr_array, ...]
) -> None:
    with pytest.raises(ValueError, match="zarr-only"):
        write_csr_matrix_group(
            dataset_format,
            tmp_path / "g",
            distinct_patterns,
            member_path="matrix",
        )


def test_accumulator_sums_duplicate_coo_entries_without_densifying() -> None:
    accumulator = CsrAccumulator()
    accumulator.append_sparse_components(
        indices=np.array([[0, 0, 1], [1, 1, 2]]),
        values=np.array([1.0, 2.0, 3.0]),
        size=(2, 3),
        repeats=1,
    )
    (sample,) = accumulator.samples
    assert sample.indices.shape == (2,)
    assert sample[0, 1] == pytest.approx(3.0)
    assert sample[1, 2] == pytest.approx(3.0)


def test_legacy_dense_manifest_without_matrix_format_loads_dense(
    tmp_path: Path, rhs_and_solutions: tuple[np.ndarray, np.ndarray]
) -> None:
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    rng = np.random.default_rng(SEED)
    dense_stack = rng.standard_normal((SAMPLE_COUNT, ROWS, ROWS))
    _write_zarr_array(dataset_dir / MATRIX_DATASET_MEMBER, dense_stack)
    _save_manifest(
        dataset_dir,
        DatasetArtifact(
            path=MATRIX_DATASET_MEMBER,
            format="zarr",
            dtype="float64",
            shape=dense_stack.shape,
            n_matrix_samples=SAMPLE_COUNT,
        ),
        *rhs_and_solutions,
    )
    manifest_file = dataset_dir / DATASET_MANIFEST_FILENAME
    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    raw["matrix"].pop("matrix_format", None)
    manifest_file.write_text(json.dumps(raw), encoding="utf-8")

    assert load_dataset_manifest(dataset_dir)["matrix"].get("matrix_format") is None
    np.testing.assert_array_equal(load_matrix_dense_sample(dataset_dir, 2), dense_stack[2])
    assert load_matrix_sparse_sample(dataset_dir, 2).shape == (ROWS, ROWS)
