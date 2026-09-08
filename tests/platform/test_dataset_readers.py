from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.dataset_readers import (
    load_matrix_dense_sample,
    resolve_canonical_training_triplet,
)
from neuralls.platform.storage.manifest_io import load_dataset_manifest
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.enum_codecs import encode_row_kind_array
from neuralls.shared.types import RowKind


def _build_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
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
            normalize="none",
        ),
        str(dataset_dir),
        dataset_format="npy",
    )
    return dataset_dir


def _drop_manifest_matrix_sample_index(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["matrix_sample_index"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_resolve_canonical_training_triplet_selects_first_standard_row(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    triplet = resolve_canonical_training_triplet(dataset_dir, sample_index=None)

    assert triplet.sample_index == 0
    assert triplet.matrix_index == 0
    assert triplet.row_kind is RowKind.STANDARD
    assert triplet.matrix_binding_enforced is True
    assert triplet.matrix.shape == (2, 2)
    assert triplet.rhs.shape == (2,)
    assert triplet.lhs.shape == (2,)


def test_resolve_canonical_training_triplet_rejects_explicit_residual_row(
    tmp_path: Path,
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    np.save(
        dataset_dir / "row_kind.npy",
        encode_row_kind_array([RowKind.STANDARD, RowKind.CG_INTERNAL, RowKind.STANDARD]),
    )

    with pytest.raises(ValueError, match="CG_INTERNAL"):
        resolve_canonical_training_triplet(dataset_dir, sample_index=1)


def test_resolve_canonical_training_triplet_requires_matrix_binding(
    tmp_path: Path,
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    _drop_manifest_matrix_sample_index(dataset_dir)

    with pytest.raises(ValueError, match="matrix_sample_index"):
        resolve_canonical_training_triplet(dataset_dir, sample_index=None)


def test_resolve_canonical_training_triplet_allows_explicit_legacy_matrix_index(
    tmp_path: Path,
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    _drop_manifest_matrix_sample_index(dataset_dir)

    triplet = resolve_canonical_training_triplet(dataset_dir, sample_index=0, matrix_index=0)

    assert triplet.matrix_index == 0
    assert triplet.matrix_binding_enforced is False


def test_load_dataset_manifest_caches_repeated_calls(tmp_path: Path) -> None:
    """Two reads of the same dataset dir hit disk only once (shared across comparisons)."""
    load_dataset_manifest.cache_clear()
    dataset_dir = _build_dataset(tmp_path)

    first = load_dataset_manifest(dataset_dir)
    second = load_dataset_manifest(dataset_dir)

    assert first == second
    assert load_dataset_manifest.cache_info().hits == 1


def test_load_matrix_dense_sample_caches_repeated_calls(tmp_path: Path) -> None:
    """Two reads of the same (dataset_dir, sample_index) hit disk only once."""
    load_matrix_dense_sample.cache_clear()
    dataset_dir = _build_dataset(tmp_path)

    first = load_matrix_dense_sample(dataset_dir, sample_index=0)
    second = load_matrix_dense_sample(dataset_dir, sample_index=0)

    assert np.array_equal(first, second)
    assert load_matrix_dense_sample.cache_info().hits == 1
