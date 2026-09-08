"""Tests for fingerprint-backed staleness detection in the generate stage.

Generation and training used to answer "has this dataset changed?" two
different ways: training fingerprinted the artifacts, generation only checked
that they existed. Generation now stamps the same fingerprint into the
manifest at write time and re-checks it before skipping a regeneration, so
both stages share one definition of a stale dataset.

These tests pin that the stamped fingerprint really is
``compute_dataset_fingerprint`` over the manifest-resolved artifacts, that a
touched artifact forces regeneration, and that manifests written before this
field existed keep their original existence-only behaviour.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import (
    _dataset_already_generated,
    _fingerprinted_artifact_paths,
    build_dataset,
)
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.caching import compute_dataset_fingerprint
from neuralls.platform.storage.dataset_readers import resolve_dataset_artifacts
from neuralls.platform.storage.manifest_io import (
    read_dataset_manifest,
    save_dataset_manifest,
)
from neuralls.shared.types import DatasetFormat


def _bump_mtime(path: Path, *, seconds: float = 1.0) -> None:
    """Advance a path's mtime deterministically, without depending on wall-clock timing."""
    stat = path.stat()
    new_time = stat.st_mtime + seconds
    os.utime(path, (new_time, new_time))


@pytest.fixture
def spd_matrix_file(tmp_path: Path) -> Path:
    """A small symmetric positive-definite matrix on disk, as a generation source."""
    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )
    matrix_file = tmp_path / "matrix.txt"
    np.savetxt(matrix_file, matrix)
    return matrix_file


@pytest.fixture
def source_spec(spd_matrix_file: Path) -> SourceSpec:
    """Source reading the single on-disk matrix, with no RHS archive."""
    return SourceSpec(matrix_path=str(spd_matrix_file), rhs_path=None)


@pytest.fixture
def dataset_spec() -> DatasetSpec:
    """Deterministic single-sample dataset spec."""
    return DatasetSpec(
        mixture=MixtureSpec(
            counts={"neutral_ones": 1},
            seed=42,
            shuffle=False,
            strategy_overrides={"neutral_ones": {"samples": 1}},
        ),
        normalize="none",
    )


@pytest.fixture(params=["npy", "hdf5"])
def dataset_format(request: pytest.FixtureRequest) -> DatasetFormat:
    """Formats whose artifacts are plain files, so mtime bumps are unambiguous."""
    return request.param


@pytest.fixture
def generated_dataset_dir(
    tmp_path: Path,
    source_spec: SourceSpec,
    dataset_spec: DatasetSpec,
    dataset_format: DatasetFormat,
) -> Path:
    """A freshly generated dataset directory, written through the real pipeline."""
    dataset_dir = tmp_path / "dataset"
    build_dataset(
        source_spec,
        dataset_spec,
        str(dataset_dir),
        dataset_format=dataset_format,
    )
    return dataset_dir


def test_fresh_generation_stamps_a_fingerprint_of_its_own_artifacts(
    generated_dataset_dir: Path,
) -> None:
    """(a) The stored fingerprint equals the fingerprint of the files just written."""
    manifest = read_dataset_manifest(generated_dataset_dir)
    artifacts = resolve_dataset_artifacts(generated_dataset_dir)

    assert manifest.dataset_fingerprint is not None
    assert manifest.dataset_fingerprint == compute_dataset_fingerprint(
        _fingerprinted_artifact_paths(artifacts)
    )


def test_stamped_fingerprint_matches_the_training_side_computation(
    generated_dataset_dir: Path,
) -> None:
    """The whole point of the field: the training reuse-check computes this same value.

    Training fingerprints ``(matrix, rhs, solutions)`` resolved from the
    manifest — recomputed here in training's own argument order to prove the
    two stages agree without sharing any state beyond the artifacts.
    """
    artifacts = resolve_dataset_artifacts(generated_dataset_dir)
    training_side = compute_dataset_fingerprint(
        (artifacts.matrix.path, artifacts.rhs.path, artifacts.solutions.path)
    )

    assert read_dataset_manifest(generated_dataset_dir).dataset_fingerprint == training_side


def test_unchanged_dataset_is_reported_as_already_generated(
    generated_dataset_dir: Path,
) -> None:
    """(b) Files present and fingerprint matching means regeneration can be skipped."""
    assert _dataset_already_generated(generated_dataset_dir) is True


def test_touched_artifact_forces_regeneration_although_every_file_exists(
    generated_dataset_dir: Path,
) -> None:
    """(c) The case existence-only checking could never catch."""
    artifacts = resolve_dataset_artifacts(generated_dataset_dir)
    paths = _fingerprinted_artifact_paths(artifacts)
    assert all(path.exists() for path in paths)

    _bump_mtime(artifacts.rhs.path)

    assert all(path.exists() for path in paths)
    assert _dataset_already_generated(generated_dataset_dir) is False


def test_regeneration_restamps_a_matching_fingerprint(
    generated_dataset_dir: Path,
    source_spec: SourceSpec,
    dataset_spec: DatasetSpec,
    dataset_format: DatasetFormat,
) -> None:
    """A stale dataset does not stay stale — regenerating makes it reusable again."""
    _bump_mtime(resolve_dataset_artifacts(generated_dataset_dir).rhs.path)
    assert _dataset_already_generated(generated_dataset_dir) is False

    build_dataset(
        source_spec,
        dataset_spec,
        str(generated_dataset_dir),
        dataset_format=dataset_format,
        force=True,
    )

    assert _dataset_already_generated(generated_dataset_dir) is True


def test_manifest_without_a_fingerprint_falls_back_to_existence_only(
    generated_dataset_dir: Path,
) -> None:
    """(d) Datasets written before this field existed keep their original behaviour.

    A missing fingerprint means "nothing is known about staleness", not
    "assume stale" — so a complete pre-fingerprint dataset is still reused,
    even with an mtime bump that a stamped manifest would have rejected.
    """
    manifest = read_dataset_manifest(generated_dataset_dir)
    save_dataset_manifest(generated_dataset_dir, replace(manifest, dataset_fingerprint=None))
    _bump_mtime(resolve_dataset_artifacts(generated_dataset_dir).rhs.path)

    assert read_dataset_manifest(generated_dataset_dir).dataset_fingerprint is None
    assert _dataset_already_generated(generated_dataset_dir) is True


def test_missing_artifact_still_forces_regeneration_before_any_fingerprint_check(
    generated_dataset_dir: Path,
) -> None:
    """Existence remains the first gate — a deleted artifact never reaches the hash."""
    artifacts = resolve_dataset_artifacts(generated_dataset_dir)
    artifacts.rhs.path.unlink()

    assert _dataset_already_generated(generated_dataset_dir) is False
