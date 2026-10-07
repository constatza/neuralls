"""A streamed dense dataset appears at its final path only as one completed commit.

Interruption is injected through the SampleWriter seam in ``dense_streaming``; the fixtures
live in ``tests/composition/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from neuralls.composition.generation import dense_streaming
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.sample_writer import SampleWriter
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.storage.dataset_digest import dataset_content_digest
from neuralls.shared.types import DatasetFormat

_INTERRUPT_AFTER_BATCHES = 1
"""Fail after the first batch, so some rows are written before the stream dies."""

_PARTIAL_SUFFIX = ".partial"
_DATASET_NAME = "dataset"

type Builder = Callable[[Path], Path]


@pytest.fixture
def build_into(
    atomic_source: SourceSpec, atomic_spec: DatasetSpec, dense_format: DatasetFormat
) -> Builder:
    """Run the public streamed build into ``<root>/dataset`` and return that final path."""

    def _build(root: Path) -> Path:
        final = root / _DATASET_NAME
        build_dataset(
            atomic_source,
            atomic_spec,
            str(final),
            dataset_format=dense_format,
            force=True,
        )
        return final

    return _build


@pytest.fixture
def reference_digest(build_into: Builder, tmp_path: Path) -> str:
    """Content digest of an uninterrupted run, the target for every rerun."""
    final = build_into(tmp_path / "reference")
    return str(dataset_content_digest(final))


def test_interrupted_run_leaves_no_final_dataset(
    build_into: Builder,
    interrupt_writer: Callable[[int], None],
    tmp_path: Path,
) -> None:
    interrupt_writer(_INTERRUPT_AFTER_BATCHES)
    with pytest.raises(RuntimeError, match="injected interruption"):
        build_into(tmp_path / "run")
    final = tmp_path / "run" / _DATASET_NAME
    assert not final.exists()
    # A .partial directory is allowed to remain; it must never be the final name.


def test_rerun_after_interrupt_regenerates(
    build_into: Builder,
    interrupt_writer: Callable[[int], None],
    monkeypatch: pytest.MonkeyPatch,
    reference_digest: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    interrupt_writer(_INTERRUPT_AFTER_BATCHES)
    with pytest.raises(RuntimeError):
        build_into(root)
    monkeypatch.setattr(dense_streaming, "SampleWriter", SampleWriter)
    final = build_into(root)
    assert str(dataset_content_digest(final)) == reference_digest
    assert not (root / f"{_DATASET_NAME}{_PARTIAL_SUFFIX}").exists()


def test_no_partial_directory_after_success(build_into: Builder, tmp_path: Path) -> None:
    final = build_into(tmp_path / "run")
    assert final.is_dir()
    assert not (final.parent / f"{final.name}{_PARTIAL_SUFFIX}").exists()


def test_commit_is_single_rename(
    build_into: Builder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The final path is absent while every array and the manifest are being written.

    The manifest is the last write of the streamed path, so the seam checks the state
    right before it runs: the final name must not exist yet, and the writes must target
    the staging directory.
    """
    root = tmp_path / "run"
    final = root / _DATASET_NAME
    seen: list[tuple[Path, bool]] = []
    original_save: Callable[..., None] = dense_streaming.save_dense_stream_manifest

    def _recording_save(dataset_dir: Path, *args: object, **kwargs: object) -> None:
        seen.append((dataset_dir, final.exists()))
        original_save(dataset_dir, *args, **kwargs)

    monkeypatch.setattr(dense_streaming, "save_dense_stream_manifest", _recording_save)
    build_into(root)
    assert seen == [(root / f"{_DATASET_NAME}{_PARTIAL_SUFFIX}", False)]
    assert final.is_dir()
