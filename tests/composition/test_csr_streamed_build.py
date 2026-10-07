"""The streamed CSR build stores the same content across formats and follows the commit and stamp rules.

The streamed branch is observed by wrapping
``dataset_builder.write_csr_streamed``. Fixtures for the matrix sources and the dense
streamed spec live in ``tests/composition/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.composition.generation import dataset_builder, finalize
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.identity import StageIdentity
from neuralls.platform.storage.dataset_digest import dataset_content_digest
from neuralls.platform.storage.dataset_readers import (
    load_matrix_sparse_sample,
)
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.types import DatasetFormat, LayoutType, MatrixFormat, SparsityPattern

_INTERRUPT_AFTER_BATCHES = 1
_RENAME = "rename"
_STAMP = "stamp"
_SINGLE_MATRIX_FILE = "A_000.txt"
_CONTENT_FORMATS: tuple[DatasetFormat, ...] = ("zarr", "hdf5")


@pytest.fixture
def csr_single_source(atomic_matrix_dir: Path) -> SourceSpec:
    return SourceSpec(matrix_path=str(atomic_matrix_dir / _SINGLE_MATRIX_FILE))


@pytest.fixture
def stage_identity() -> StageIdentity:
    return StageIdentity(stage="generation", key="csr-streamed-test", components={"spec": "fixed"})


@pytest.fixture
def streamed_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every dataset directory that takes the streamed CSR branch."""
    calls: list[Path] = []
    original = dataset_builder.write_csr_streamed

    def _recording(
        source: SourceSpec,
        spec: DatasetSpec,
        dataset_dir: Path,
        dataset_format: DatasetFormat,
        *,
        sparsity_pattern: SparsityPattern,
    ) -> None:
        calls.append(dataset_dir)
        return original(
            source, spec, dataset_dir, dataset_format, sparsity_pattern=sparsity_pattern
        )

    monkeypatch.setattr(dataset_builder, "write_csr_streamed", _recording)
    return calls


@pytest.fixture
def event_log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the commit rename and the identity stamp in the order they happen."""
    events: list[str] = []
    original_rename = finalize._rename_staged
    original_stamp: Callable[..., None] = dataset_builder._stamp_dataset_identity

    def _recording_rename(staging: Path, final: Path) -> None:
        events.append(_RENAME)
        original_rename(staging, final)

    def _recording_stamp(dataset_dir: Path, *args: object, **kwargs: object) -> None:
        events.append(_STAMP)
        original_stamp(dataset_dir, *args, **kwargs)

    monkeypatch.setattr(finalize, "_rename_staged", _recording_rename)
    monkeypatch.setattr(dataset_builder, "_stamp_dataset_identity", _recording_stamp)
    return events


@pytest.fixture
def interrupt_csr_stream(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Make the streamed CSR sink raise on the batch after ``n`` batches were written."""

    def _install(after_batches: int) -> None:
        def _open(
            dataset_format: DatasetFormat,
            container: Path,
            *,
            planned_samples: int,
            layout: LayoutType,
        ) -> object:
            from neuralls.platform.storage import csr_storage

            inner = csr_storage.open_csr_matrix_stream(
                dataset_format, container, planned_samples=planned_samples, layout=layout
            )
            written = 0

            class _InterruptedStream:
                def write_batch(self, samples: Sequence[csr_array]) -> None:
                    nonlocal written
                    if written >= after_batches:
                        raise RuntimeError("injected interruption")
                    written += 1
                    inner.write_batch(samples)

                def close(self) -> object:
                    return inner.close()

            return _InterruptedStream()

        monkeypatch.setattr(
            "neuralls.composition.generation.csr_streaming.open_csr_matrix_stream", _open
        )

    return _install


def _streamed_digests(source: SourceSpec, spec: DatasetSpec, root: Path) -> dict[str, str]:
    """Content digest of the same streamed CSR build, written once per storage format.

    Both formats go through ``write_csr_streamed``; only the container differs. The
    content digest hashes the stored arrays, so equal digests mean equal stored content.
    """
    digests: dict[str, str] = {}
    for dataset_format in _CONTENT_FORMATS:
        dataset_dir = root / dataset_format
        build_dataset(
            source,
            spec,
            str(dataset_dir),
            dataset_format=dataset_format,
            matrix_format=MatrixFormat.CSR,
            sparsity_pattern=SparsityPattern.RAGGED,
        )
        digests[dataset_format] = dataset_content_digest(dataset_dir)
    return digests


def test_csr_streamed_content_digest_matches_across_formats_single_matrix(
    csr_single_source: SourceSpec,
    atomic_spec: DatasetSpec,
    tmp_path: Path,
) -> None:
    """One matrix: zarr and hdf5 hold the same streamed CSR content, so the digests agree."""
    digests = _streamed_digests(csr_single_source, atomic_spec, tmp_path)

    assert set(digests) == set(_CONTENT_FORMATS)
    assert len(set(digests.values())) == 1


def test_csr_streamed_content_digest_matches_across_formats_several_matrices(
    atomic_source: SourceSpec,
    atomic_spec: DatasetSpec,
    tmp_path: Path,
) -> None:
    """Several matrices: the per-sample ragged layout is identical in zarr and hdf5."""
    digests = _streamed_digests(atomic_source, atomic_spec, tmp_path)

    assert set(digests) == set(_CONTENT_FORMATS)
    assert len(set(digests.values())) == 1


def test_csr_build_has_no_final_directory_until_complete(
    atomic_source: SourceSpec,
    atomic_spec: DatasetSpec,
    dense_format: DatasetFormat,
    interrupt_csr_stream: Callable[[int], None],
    tmp_path: Path,
) -> None:
    interrupt_csr_stream(_INTERRUPT_AFTER_BATCHES)
    final = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="injected interruption"):
        build_dataset(
            atomic_source,
            atomic_spec,
            str(final),
            dataset_format=dense_format,
            matrix_format=MatrixFormat.CSR,
        )
    assert not final.exists()


def test_csr_build_identity_stamped_after_commit(
    atomic_source: SourceSpec,
    atomic_spec: DatasetSpec,
    dense_format: DatasetFormat,
    stage_identity: StageIdentity,
    event_log: list[str],
    tmp_path: Path,
) -> None:
    final = tmp_path / "dataset"
    build_dataset(
        atomic_source,
        atomic_spec,
        str(final),
        dataset_format=dense_format,
        identity=stage_identity,
        matrix_format=MatrixFormat.CSR,
    )
    assert event_log == [_RENAME, _STAMP]
    manifest = (final / DATASET_MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert stage_identity.key in manifest


def test_csr_one_and_several_matrices_use_ragged_layout_by_default(
    csr_single_source: SourceSpec,
    atomic_source: SourceSpec,
    atomic_spec: DatasetSpec,
    dense_format: DatasetFormat,
    tmp_path: Path,
) -> None:
    one = tmp_path / "one"
    several = tmp_path / "several"
    build_dataset(
        csr_single_source,
        atomic_spec,
        str(one),
        dataset_format=dense_format,
        matrix_format=MatrixFormat.CSR,
    )
    build_dataset(
        atomic_source,
        atomic_spec,
        str(several),
        dataset_format=dense_format,
        matrix_format=MatrixFormat.CSR,
    )
    assert read_dataset_manifest(one).matrix.layout is LayoutType.MANY_MATRICES
    assert read_dataset_manifest(several).matrix.layout is LayoutType.MANY_MATRICES


@pytest.mark.parametrize("dataset_format", _CONTENT_FORMATS)
def test_csr_streamed_round_trip_returns_the_source_matrix(
    dataset_format: DatasetFormat,
    csr_single_source: SourceSpec,
    atomic_spec: DatasetSpec,
    atomic_matrix_dir: Path,
    tmp_path: Path,
) -> None:
    """Reading the stored sparse sample back gives the dense source matrix, in each format."""
    dataset_dir = tmp_path / dataset_format
    unnormalized = replace(atomic_spec, normalize="none")
    build_dataset(
        csr_single_source,
        unnormalized,
        str(dataset_dir),
        dataset_format=dataset_format,
        matrix_format=MatrixFormat.CSR,
    )
    source_matrix = np.loadtxt(atomic_matrix_dir / _SINGLE_MATRIX_FILE)

    stored = load_matrix_sparse_sample(dataset_dir, 0).toarray()

    np.testing.assert_array_equal(stored, source_matrix)
