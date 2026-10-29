"""Tests for path-preserving training artifact resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest
from neuralls.platform.storage.training_artifacts import (
    NpyArraySource,
    ZarrArraySource,
    load_array_source_sample,
    load_training_arrays,
    parameters_zarr_paths,
)
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX


@pytest.fixture
def minimal_dataset(tmp_path: Path) -> Path:
    """Dataset with rhs.zarr, solutions.zarr, and matrix.zarr but no parameters."""
    n, samples = 4, 6
    for name in ("rhs.zarr", "solutions.zarr", "matrix.zarr"):
        store = tmp_path / name
        store.mkdir()
        (store / "zarr.json").write_text(
            '{"zarr_format": 3, "node_type": "array"}', encoding="utf-8"
        )
    save_dataset_manifest(
        tmp_path,
        make_dataset_manifest(
            matrix=DatasetArtifact(
                path="matrix.zarr",
                format="zarr",
                dtype="float64",
                shape=(1, n, n),
                n_matrix_samples=1,
                broadcast=True,
            ),
            rhs=DatasetArtifact(
                path="rhs.zarr", format="zarr", dtype="float64", shape=(samples, n)
            ),
            solutions=DatasetArtifact(
                path="solutions.zarr",
                format="zarr",
                dtype="float64",
                shape=(samples, n),
            ),
            normalization=DatasetNormalization(
                type="none",
                matrix_norm=1.0,
                matrix_norm_type="fro",
                scale={},
            ),
        ),
    )
    return tmp_path


@pytest.fixture
def two_parameters_zarrs(minimal_dataset: Path) -> Path:
    """Adds parameters_0.zarr and parameters_1.zarr to the minimal dataset."""
    samples, p = 6, 3
    params: list[DatasetArtifact] = []
    for i in range(2):
        store = minimal_dataset / f"{PARAMETERS_ZARR_PREFIX}{i}.zarr"
        store.mkdir()
        (store / "zarr.json").write_text(
            '{"zarr_format": 3, "node_type": "array"}', encoding="utf-8"
        )
        params.append(
            DatasetArtifact(
                path=f"{PARAMETERS_ZARR_PREFIX}{i}.zarr",
                format="zarr",
                dtype="float64",
                shape=(samples, p),
            )
        )
    save_dataset_manifest(
        minimal_dataset,
        make_dataset_manifest(
            matrix=DatasetArtifact(
                path="matrix.zarr",
                format="zarr",
                dtype="float64",
                shape=(1, 4, 4),
                n_matrix_samples=1,
                broadcast=True,
            ),
            rhs=DatasetArtifact(
                path="rhs.zarr", format="zarr", dtype="float64", shape=(samples, 4)
            ),
            solutions=DatasetArtifact(
                path="solutions.zarr",
                format="zarr",
                dtype="float64",
                shape=(samples, 4),
            ),
            normalization=DatasetNormalization(
                type="none",
                matrix_norm=1.0,
                matrix_norm_type="fro",
                scale={},
            ),
            params=tuple(params),
        ),
    )
    return minimal_dataset


def test_load_training_arrays_detects_parameters_zarr(two_parameters_zarrs: Path) -> None:
    arrays = load_training_arrays(two_parameters_zarrs)

    assert len(arrays.parameter_sources) == 2
    assert isinstance(arrays.parameter_sources[0], ZarrArraySource)
    assert parameters_zarr_paths(arrays)[0].stem == f"{PARAMETERS_ZARR_PREFIX}0"
    assert parameters_zarr_paths(arrays)[1].stem == f"{PARAMETERS_ZARR_PREFIX}1"


def test_load_training_arrays_returns_empty_parameters_when_absent(
    minimal_dataset: Path,
) -> None:
    arrays = load_training_arrays(minimal_dataset)
    assert arrays.parameter_sources == ()


def test_load_training_arrays_parameters_sorted_by_index(two_parameters_zarrs: Path) -> None:
    """parameters_zarr must be ordered by index regardless of filesystem traversal order."""
    arrays = load_training_arrays(two_parameters_zarrs)
    stems = [p.stem for p in parameters_zarr_paths(arrays)]
    assert stems == sorted(stems)


def test_load_training_arrays_preserves_npy_rhs_and_solution_paths(tmp_path: Path) -> None:
    """Numpy artifacts remain path-backed instead of being eagerly loaded."""
    rng = np.random.default_rng(2)
    rhs = rng.standard_normal((5, 4))
    solutions = rng.standard_normal((5, 4))
    np.save(tmp_path / "rhs.npy", rhs)
    np.save(tmp_path / "solutions.npy", solutions)
    matrix_store = tmp_path / "matrix.zarr"
    matrix_store.mkdir()
    (matrix_store / "zarr.json").write_text(
        '{"zarr_format": 3, "node_type": "array"}', encoding="utf-8"
    )
    save_dataset_manifest(
        tmp_path,
        make_dataset_manifest(
            matrix=DatasetArtifact(
                path="matrix.zarr",
                format="zarr",
                dtype="float64",
                shape=(1, 4, 4),
                n_matrix_samples=1,
                broadcast=True,
            ),
            rhs=DatasetArtifact(path="rhs.npy", format="npy", dtype="float64", shape=(5, 4)),
            solutions=DatasetArtifact(
                path="solutions.npy", format="npy", dtype="float64", shape=(5, 4)
            ),
            normalization=DatasetNormalization(
                type="none",
                matrix_norm=1.0,
                matrix_norm_type="fro",
                scale={},
            ),
        ),
    )

    arrays = load_training_arrays(tmp_path)

    assert isinstance(arrays.rhs_source, NpyArraySource)
    assert isinstance(arrays.solutions_source, NpyArraySource)
    assert arrays.rhs_source.path == tmp_path / "rhs.npy"
    assert arrays.solutions_source.path == tmp_path / "solutions.npy"
    assert arrays.sample_count == 5


def test_load_array_source_sample_supports_npy(tmp_path: Path) -> None:
    """Single-sample extraction supports numpy-backed sources via mmap."""
    data = np.arange(12.0).reshape(3, 4)
    path = tmp_path / "rhs.npy"
    np.save(path, data)

    sample = load_array_source_sample(NpyArraySource(path=path), sample_index=1)

    np.testing.assert_array_equal(sample, data[1])
