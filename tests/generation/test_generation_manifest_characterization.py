"""Characterization tests pinning the on-disk manifest of every storage format.

These tests assert the exact serialized manifest dictionary written by each
``GenerationDatasetStorage`` implementation so that refactors of the shared
manifest-assembly logic cannot silently change observable output.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest
import zarr

from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.platform.storage.datasets import make_generation_dataset_storage
from neuralls.platform.storage.manifest import manifest_path_for
from neuralls.shared.types import DatasetFormat, LayoutType, ScaleMetadata

ALL_FORMATS: tuple[DatasetFormat, ...] = ("zarr", "npy", "hdf5")


@pytest.fixture(params=ALL_FORMATS)
def dataset_format(request: pytest.FixtureRequest) -> DatasetFormat:
    """Every supported generation storage format."""
    return cast(DatasetFormat, request.param)


@pytest.fixture
def many_matrices_array() -> np.ndarray:
    """Three distinct 2x2 matrices, one per logical sample."""
    return np.array(
        [
            [[4.0, 1.0], [1.0, 4.0]],
            [[5.0, 2.0], [2.0, 5.0]],
            [[6.0, 3.0], [3.0, 6.0]],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def broadcast_matrix_array() -> np.ndarray:
    """A single 2x2 matrix shared by every logical sample."""
    return np.array([[[9.0, 0.0], [0.0, 9.0]]], dtype=np.float64)


@pytest.fixture
def rhs_array() -> np.ndarray:
    """RHS vectors for three logical samples."""
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)


@pytest.fixture
def solutions_array() -> np.ndarray:
    """Solution vectors aligned with ``rhs_array``."""
    return np.array([[0.25, 0.5], [0.75, 1.0], [1.25, 1.5]], dtype=np.float64)


@pytest.fixture
def row_kind_codes() -> np.ndarray:
    """Semantic row-kind codes aligned with the persisted rows."""
    return np.array([0, 1, 2], dtype=np.uint8)


@pytest.fixture
def matrix_sample_index() -> np.ndarray:
    """Per-row physical matrix binding aligned with the persisted rows."""
    return np.array([0, 1, 2], dtype=np.int64)


@pytest.fixture
def parameters_arrays() -> tuple[np.ndarray, ...]:
    """Parameter arrays including an empty entry that must be skipped."""
    return (
        np.array([[1.5], [2.5], [3.5]], dtype=np.float64),
        np.zeros((0, 0), dtype=np.float64),
        np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float64),
    )


@pytest.fixture
def scale_metadata() -> ScaleMetadata:
    """Normalization scale metadata carried into the manifest."""
    return ScaleMetadata(spectral_radius_bound=2.5, dimension_scale=0.5)


@pytest.fixture
def stage_matrix(tmp_path: Path) -> Callable[[DatasetFormat, str, np.ndarray], Path]:
    """Write a staged matrix artifact in the layout each storage backend expects."""

    def stage(dataset_format: DatasetFormat, name: str, matrix: np.ndarray) -> Path:
        staging_path = tmp_path / f"{name}-staging"
        match dataset_format:
            case "hdf5":
                h5_path = staging_path.with_suffix(".h5")
                with h5py.File(str(h5_path), "w") as out:
                    out.create_dataset("matrix", data=matrix)
                return h5_path
            case "npy":
                npy_path = staging_path.with_suffix(".npy")
                np.save(npy_path, matrix)
                return npy_path
            case "zarr":
                zarr_path = staging_path.with_suffix(".zarr")
                array = zarr.open_array(
                    str(zarr_path),
                    mode="w",
                    shape=matrix.shape,
                    chunks=(1, *matrix.shape[1:]),
                    dtype="float64",
                )
                array[:] = matrix
                return zarr_path

    return stage


@pytest.fixture
def full_payload_factory(
    stage_matrix: Callable[[DatasetFormat, str, np.ndarray], Path],
    many_matrices_array: np.ndarray,
    rhs_array: np.ndarray,
    solutions_array: np.ndarray,
    row_kind_codes: np.ndarray,
    matrix_sample_index: np.ndarray,
    parameters_arrays: tuple[np.ndarray, ...],
    scale_metadata: ScaleMetadata,
) -> Callable[[DatasetFormat], GeneratedDatasetPayload]:
    """Build a many-matrices payload with every optional artifact populated."""

    def make(dataset_format: DatasetFormat) -> GeneratedDatasetPayload:
        return GeneratedDatasetPayload(
            rhs=rhs_array,
            solutions=solutions_array,
            matrix_artifact_path=stage_matrix(dataset_format, "full", many_matrices_array),
            matrix_size=(2, 2),
            normalization_type="spectral",
            matrix_norm=3.5,
            matrix_norm_type="spectral_radius",
            scale_metadata=scale_metadata,
            parameters_arrays=parameters_arrays,
            layout=LayoutType.MANY_MATRICES,
            row_kind_codes=row_kind_codes,
            matrix_sample_index=matrix_sample_index,
        )

    return make


@pytest.fixture
def minimal_payload_factory(
    stage_matrix: Callable[[DatasetFormat, str, np.ndarray], Path],
    broadcast_matrix_array: np.ndarray,
    rhs_array: np.ndarray,
    solutions_array: np.ndarray,
) -> Callable[[DatasetFormat], GeneratedDatasetPayload]:
    """Build a broadcast-single payload with no optional artifacts."""

    def make(dataset_format: DatasetFormat) -> GeneratedDatasetPayload:
        return GeneratedDatasetPayload(
            rhs=rhs_array,
            solutions=solutions_array,
            matrix_artifact_path=stage_matrix(dataset_format, "minimal", broadcast_matrix_array),
            matrix_size=(2, 2),
            normalization_type="none",
            matrix_norm=1.0,
            matrix_norm_type="frobenius",
            layout=LayoutType.BROADCAST_SINGLE,
        )

    return make


def _write_and_read_manifest(
    dataset_dir: Path,
    dataset_format: DatasetFormat,
    payload: GeneratedDatasetPayload,
) -> dict[str, Any]:
    storage = make_generation_dataset_storage(dataset_format)
    storage.write_dataset(dataset_dir, payload)
    return json.loads(manifest_path_for(dataset_dir).read_text(encoding="utf-8"))


def _expected_full_manifest(dataset_format: DatasetFormat) -> dict[str, Any]:
    """Expected manifest for the fully-populated many-matrices payload."""
    normalization = {
        "type": "spectral",
        "matrix_norm": 3.5,
        "matrix_norm_type": "spectral_radius",
        "scale": {"spectral_radius_bound": 2.5, "dimension_scale": 0.5},
    }
    matrix_extras = {
        "n_matrix_samples": 3,
        "broadcast": False,
        "layout": "many_matrices",
        "logical_sample_count": 3,
        "index": None,
    }
    match dataset_format:
        case "zarr":
            prefix = "dataset.zarr/"
            return {
                "schema": "neuralls.dataset.v2",
                "matrix": {
                    "path": f"{prefix}matrix",
                    "format": "zarr",
                    "dtype": "float64",
                    "shape": [3, 2, 2],
                    "key": None,
                    **matrix_extras,
                },
                "rhs": _expected_vector(f"{prefix}rhs", "zarr", None),
                "solutions": _expected_vector(f"{prefix}solutions", "zarr", None),
                "normalization": normalization,
                "params": [
                    _expected_param(f"{prefix}parameters_0", "zarr", None, 0, [3, 1]),
                    _expected_param(f"{prefix}parameters_2", "zarr", None, 2, [3, 2]),
                ],
                "row_kind": _expected_optional(f"{prefix}row_kind", "zarr", None, "uint8", [3]),
                "matrix_sample_index": _expected_optional(
                    f"{prefix}matrix_sample_index", "zarr", None, "int64", [3]
                ),
                "dataset_fingerprint": None,
            }
        case "npy":
            return {
                "schema": "neuralls.dataset.v2",
                "matrix": {
                    "path": "matrix.npy",
                    "format": "npy",
                    "dtype": "float64",
                    "shape": [3, 2, 2],
                    "key": None,
                    **matrix_extras,
                },
                "rhs": _expected_vector("rhs.npy", "npy", None),
                "solutions": _expected_vector("solutions.npy", "npy", None),
                "normalization": normalization,
                "params": [
                    _expected_param("parameters_0.npy", "npy", None, 0, [3, 1]),
                    _expected_param("parameters_2.npy", "npy", None, 2, [3, 2]),
                ],
                "row_kind": _expected_optional("row_kind.npy", "npy", None, "uint8", [3]),
                "matrix_sample_index": _expected_optional(
                    "matrix_sample_index.npy", "npy", None, "int64", [3]
                ),
                "dataset_fingerprint": None,
            }
        case "hdf5":
            name = "dataset.h5"
            return {
                "schema": "neuralls.dataset.v2",
                "matrix": {
                    "path": name,
                    "format": "hdf5",
                    "dtype": "float64",
                    "shape": [3, 2, 2],
                    "key": "matrix",
                    **matrix_extras,
                },
                "rhs": _expected_vector(name, "hdf5", "rhs"),
                "solutions": _expected_vector(name, "hdf5", "solutions"),
                "normalization": normalization,
                "params": [
                    _expected_param(name, "hdf5", "parameters_0", 0, [3, 1]),
                    _expected_param(name, "hdf5", "parameters_2", 2, [3, 2]),
                ],
                "row_kind": _expected_optional(name, "hdf5", "row_kind", "uint8", [3]),
                "matrix_sample_index": _expected_optional(
                    name, "hdf5", "matrix_sample_index", "int64", [3]
                ),
                "dataset_fingerprint": None,
            }
        case _:  # pragma: no cover - guards fixture drift
            raise AssertionError(f"unhandled format {dataset_format!r}")


def _expected_minimal_manifest(dataset_format: DatasetFormat) -> dict[str, Any]:
    """Expected manifest for the broadcast-single payload with no optionals."""
    normalization = {
        "type": "none",
        "matrix_norm": 1.0,
        "matrix_norm_type": "frobenius",
        "scale": {},
    }
    matrix_extras = {
        "n_matrix_samples": 1,
        "broadcast": True,
        "layout": "broadcast_single",
        "logical_sample_count": 3,
        "index": None,
    }
    match dataset_format:
        case "zarr":
            prefix = "dataset.zarr/"
            matrix_path, matrix_key = f"{prefix}matrix", None
            rhs_path, rhs_key = f"{prefix}rhs", None
            solutions_path, solutions_key = f"{prefix}solutions", None
        case "npy":
            matrix_path, matrix_key = "matrix.npy", None
            rhs_path, rhs_key = "rhs.npy", None
            solutions_path, solutions_key = "solutions.npy", None
        case "hdf5":
            matrix_path, matrix_key = "dataset.h5", "matrix"
            rhs_path, rhs_key = "dataset.h5", "rhs"
            solutions_path, solutions_key = "dataset.h5", "solutions"
        case _:  # pragma: no cover - guards fixture drift
            raise AssertionError(f"unhandled format {dataset_format!r}")
    return {
        "schema": "neuralls.dataset.v2",
        "matrix": {
            "path": matrix_path,
            "format": dataset_format,
            "dtype": "float64",
            "shape": [1, 2, 2],
            "key": matrix_key,
            **matrix_extras,
        },
        "rhs": _expected_vector(rhs_path, dataset_format, rhs_key),
        "solutions": _expected_vector(solutions_path, dataset_format, solutions_key),
        "normalization": normalization,
        "params": None,
        "row_kind": None,
        "matrix_sample_index": None,
        "dataset_fingerprint": None,
    }


def _expected_vector(path: str, dataset_format: str, key: str | None) -> dict[str, Any]:
    return {
        "path": path,
        "format": dataset_format,
        "dtype": "float64",
        "shape": [3, 2],
        "index": None,
        "n_matrix_samples": None,
        "broadcast": None,
        "key": key,
        "layout": None,
        "logical_sample_count": None,
    }


def _expected_param(
    path: str,
    dataset_format: str,
    key: str | None,
    index: int,
    shape: list[int],
) -> dict[str, Any]:
    return {
        "path": path,
        "format": dataset_format,
        "dtype": "float64",
        "shape": shape,
        "index": index,
        "n_matrix_samples": None,
        "broadcast": None,
        "key": key,
        "layout": None,
        "logical_sample_count": None,
    }


def _expected_optional(
    path: str,
    dataset_format: str,
    key: str | None,
    dtype: str,
    shape: list[int],
) -> dict[str, Any]:
    return {
        "path": path,
        "format": dataset_format,
        "dtype": dtype,
        "shape": shape,
        "index": None,
        "n_matrix_samples": None,
        "broadcast": None,
        "key": key,
        "layout": None,
        "logical_sample_count": None,
    }


def test_full_payload_manifest_matches_expected_shape(
    tmp_path: Path,
    dataset_format: DatasetFormat,
    full_payload_factory: Callable[[DatasetFormat], GeneratedDatasetPayload],
) -> None:
    manifest = _write_and_read_manifest(
        tmp_path / f"{dataset_format}-full",
        dataset_format,
        full_payload_factory(dataset_format),
    )
    assert manifest == _expected_full_manifest(dataset_format)


def test_minimal_payload_manifest_matches_expected_shape(
    tmp_path: Path,
    dataset_format: DatasetFormat,
    minimal_payload_factory: Callable[[DatasetFormat], GeneratedDatasetPayload],
) -> None:
    manifest = _write_and_read_manifest(
        tmp_path / f"{dataset_format}-minimal",
        dataset_format,
        minimal_payload_factory(dataset_format),
    )
    assert manifest == _expected_minimal_manifest(dataset_format)


def test_manifest_is_byte_stable_across_repeated_writes(
    tmp_path: Path,
    dataset_format: DatasetFormat,
    full_payload_factory: Callable[[DatasetFormat], GeneratedDatasetPayload],
) -> None:
    dataset_dir = tmp_path / f"{dataset_format}-stable"
    storage = make_generation_dataset_storage(dataset_format)
    storage.write_dataset(dataset_dir, full_payload_factory(dataset_format))
    first = manifest_path_for(dataset_dir).read_bytes()
    storage.write_dataset(dataset_dir, full_payload_factory(dataset_format))
    assert manifest_path_for(dataset_dir).read_bytes() == first


def test_normalization_block_is_identical_across_formats(
    tmp_path: Path,
    full_payload_factory: Callable[[DatasetFormat], GeneratedDatasetPayload],
) -> None:
    blocks = [
        _write_and_read_manifest(tmp_path / f"{fmt}-norm", fmt, full_payload_factory(fmt))[
            "normalization"
        ]
        for fmt in ALL_FORMATS
    ]
    assert blocks[0] == blocks[1] == blocks[2]
