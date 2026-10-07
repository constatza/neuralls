"""Replacing an existing dataset directory is atomic: the old dataset survives any failed swap.

A forced regeneration renames the old dataset aside, renames the staged dataset into place,
and only then deletes the old copy. The fixtures live in ``tests/composition/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import h5py
import pytest

from neuralls.composition.generation import dense_streaming, finalize
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.storage.dataset_digest import dataset_content_digest
from neuralls.shared.types import DatasetFormat

_DATASET_NAME = "dataset"
_OLD_SUFFIX = ".old"
_PARTIAL_SUFFIX = ".partial"
_REPLACEMENT_SEED = 7
"""Seed of the regenerated dataset; differs from the fixture seed so the content changes."""

type Builder = Callable[[DatasetSpec], Path]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Per-test output root that holds the dataset directory."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    return run_root


@pytest.fixture
def replace_build(atomic_source: SourceSpec, dense_format: DatasetFormat, root: Path) -> Builder:
    """Forced streamed build into ``<root>/dataset``; the spec selects the content."""

    def _build(spec: DatasetSpec) -> Path:
        final = root / _DATASET_NAME
        build_dataset(atomic_source, spec, str(final), dataset_format=dense_format, force=True)
        return final

    return _build


@pytest.fixture
def replacement_spec(atomic_spec: DatasetSpec) -> DatasetSpec:
    return replace(atomic_spec, mixture=replace(atomic_spec.mixture, seed=_REPLACEMENT_SEED))


def test_forced_regeneration_replaces_old_dataset(
    replace_build: Builder,
    atomic_spec: DatasetSpec,
    replacement_spec: DatasetSpec,
    root: Path,
) -> None:
    replace_build(atomic_spec)
    old_digest = dataset_content_digest(root / _DATASET_NAME)

    final = replace_build(replacement_spec)

    assert final.is_dir()
    assert dataset_content_digest(final) != old_digest
    assert not (root / f"{_DATASET_NAME}{_OLD_SUFFIX}").exists()
    assert not (root / f"{_DATASET_NAME}{_PARTIAL_SUFFIX}").exists()


def test_failed_replacement_restores_old_dataset(
    replace_build: Builder,
    atomic_spec: DatasetSpec,
    replacement_spec: DatasetSpec,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replace_build(atomic_spec)
    old_digest = dataset_content_digest(root / _DATASET_NAME)
    final = root / _DATASET_NAME
    staged = finalize.staging_dir_for(final)
    original_replace = finalize.os.replace

    def _fail_staged_to_final(src: str | Path, dst: str | Path) -> None:
        if Path(src) == staged:
            raise OSError("injected failure at staged-to-final rename")
        original_replace(src, dst)

    monkeypatch.setattr(finalize.os, "replace", _fail_staged_to_final)
    with pytest.raises(OSError, match="Committing dataset"):
        replace_build(replacement_spec)
    monkeypatch.undo()

    assert final.is_dir()
    assert dataset_content_digest(final) == old_digest
    assert not (root / f"{_DATASET_NAME}{_OLD_SUFFIX}").exists()


def test_stale_old_directory_is_removed_before_replacement(
    replace_build: Builder,
    atomic_spec: DatasetSpec,
    replacement_spec: DatasetSpec,
    root: Path,
) -> None:
    replace_build(atomic_spec)
    stale = root / f"{_DATASET_NAME}{_OLD_SUFFIX}"
    stale.mkdir()
    (stale / "leftover.txt").write_text("from an earlier crash")

    final = replace_build(replacement_spec)

    assert final.is_dir()
    assert not stale.exists()


def _open_hdf5_file_count() -> int:
    """Number of HDF5 files open in this process; a finished writer must release its file."""
    return h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE)


def test_handles_are_closed_before_staged_rename(
    replace_build: Builder,
    replacement_spec: DatasetSpec,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staged rename runs only after the writer has returned and released its file."""
    baseline = _open_hdf5_file_count()
    open_at_rename: list[int] = []
    original_rename = finalize._rename_staged

    def _recording_rename(staging: Path, final: Path) -> None:
        open_at_rename.append(_open_hdf5_file_count())
        original_rename(staging, final)

    monkeypatch.setattr(finalize, "_rename_staged", _recording_rename)
    final = replace_build(replacement_spec)

    assert open_at_rename == [baseline]
    assert final.is_dir()


def test_leftover_old_is_restored_when_final_is_missing(
    replace_build: Builder,
    atomic_spec: DatasetSpec,
    replacement_spec: DatasetSpec,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after the move-aside leaves only ``.old``; the next commit restores it first."""
    replace_build(atomic_spec)
    final = root / _DATASET_NAME
    aside = root / f"{_DATASET_NAME}{_OLD_SUFFIX}"
    old_digest = dataset_content_digest(final)
    final.rename(aside)
    seen_before_write: list[tuple[bool, str]] = []
    original_save: Callable[..., None] = dense_streaming.save_dense_stream_manifest

    def _recording_save(dataset_dir: Path, *args: object, **kwargs: object) -> None:
        seen_before_write.append(
            (aside.exists(), str(dataset_content_digest(final))) if final.exists() else (False, "")
        )
        original_save(dataset_dir, *args, **kwargs)

    monkeypatch.setattr(dense_streaming, "save_dense_stream_manifest", _recording_save)
    replace_build(replacement_spec)

    assert seen_before_write == [(False, str(old_digest))]
    assert dataset_content_digest(final) != old_digest
    assert not aside.exists()


def test_permission_error_on_rename_is_retried(
    replace_build: Builder,
    replacement_spec: DatasetSpec,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient PermissionError (Windows antivirus or indexing) succeeds on a later try."""
    final = root / _DATASET_NAME
    staged = finalize.staging_dir_for(final)
    sleeps: list[float] = []
    failures_left = [2]
    original_replace = finalize.os.replace

    def _flaky_replace(src: str | Path, dst: str | Path) -> None:
        if Path(src) == staged and failures_left[0] > 0:
            failures_left[0] -= 1
            raise PermissionError("injected transient lock")
        original_replace(src, dst)

    monkeypatch.setattr(finalize.os, "replace", _flaky_replace)
    monkeypatch.setattr(finalize.time, "sleep", sleeps.append)
    replace_build(replacement_spec)

    assert failures_left == [0]
    assert sleeps == [finalize.RENAME_RETRY_DELAY_SECONDS] * 2
    assert final.is_dir()
    assert not staged.exists()


def test_persistent_permission_error_restores_old_dataset(
    replace_build: Builder,
    atomic_spec: DatasetSpec,
    replacement_spec: DatasetSpec,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replace_build(atomic_spec)
    final = root / _DATASET_NAME
    old_digest = dataset_content_digest(final)
    staged = finalize.staging_dir_for(final)
    attempts: list[Path] = []
    original_replace = finalize.os.replace

    def _locked_replace(src: str | Path, dst: str | Path) -> None:
        if Path(src) == staged:
            attempts.append(Path(src))
            raise PermissionError("injected persistent lock")
        original_replace(src, dst)

    monkeypatch.setattr(finalize.os, "replace", _locked_replace)
    monkeypatch.setattr(finalize.time, "sleep", lambda _: None)
    with pytest.raises(OSError, match="Committing dataset"):
        replace_build(replacement_spec)

    assert len(attempts) == finalize.RENAME_ATTEMPTS
    assert dataset_content_digest(final) == old_digest


def test_other_os_error_is_not_retried(
    replace_build: Builder,
    replacement_spec: DatasetSpec,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = finalize.staging_dir_for(root / _DATASET_NAME)
    attempts: list[Path] = []
    sleeps: list[float] = []
    original_replace = finalize.os.replace

    def _broken_replace(src: str | Path, dst: str | Path) -> None:
        if Path(src) == staged:
            attempts.append(Path(src))
            raise OSError("injected non-transient failure")
        original_replace(src, dst)

    monkeypatch.setattr(finalize.os, "replace", _broken_replace)
    monkeypatch.setattr(finalize.time, "sleep", sleeps.append)
    with pytest.raises(OSError, match="Committing dataset"):
        replace_build(replacement_spec)

    assert len(attempts) == 1
    assert sleeps == []
