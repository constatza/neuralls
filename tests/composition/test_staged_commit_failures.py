"""Failures while committing a staged dataset directory must not hide the restore outcome."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage import staged_commit as finalize


class _CommitFailure(OSError):
    """Stands in for the staged rename failing."""


class _RestoreFailure(OSError):
    """Stands in for moving the previous dataset back after a failed rename."""


@pytest.fixture
def existing_dataset(tmp_path: Path) -> Path:
    """A final dataset directory that a forced replacement would move aside."""
    final_dir = tmp_path / "dataset"
    final_dir.mkdir()
    (final_dir / "old.txt").write_text("previous", encoding="utf-8")
    return final_dir


@pytest.fixture
def staged_dir(tmp_path: Path) -> Path:
    """A staging directory holding the new dataset, as ``commit_staged_directory`` leaves it."""
    staging = tmp_path / "dataset.partial"
    staging.mkdir()
    (staging / "new.txt").write_text("replacement", encoding="utf-8")
    return staging


def test_failed_restore_chains_the_original_commit_error(
    existing_dataset: Path,
    staged_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both the commit and the restore fail, the restore error is raised with the cause."""
    real_move = finalize._move

    def failing_commit_move(source: Path, destination: Path) -> None:
        if source == staged_dir:
            raise _CommitFailure()
        if source == finalize.aside_dir_for(existing_dataset):
            raise _RestoreFailure()
        real_move(source, destination)

    monkeypatch.setattr(finalize, "_move", failing_commit_move)
    with pytest.raises(_RestoreFailure) as excinfo:
        finalize._rename_staged(staged_dir, existing_dataset)
    assert isinstance(excinfo.value.__cause__, _CommitFailure)


def test_corrupt_manifest_in_target_directory_is_not_overwritten(tmp_path: Path) -> None:
    """A manifest that cannot be parsed must stop the build instead of being replaced."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError):
        build_dataset(
            SourceSpec(matrix_path=str(tmp_path / "missing.npy")),
            DatasetSpec(
                mixture=MixtureSpec(counts={"gaussian_forward": 1}, seed=0, shuffle=False),
                normalize="none",
            ),
            str(dataset_dir),
            dataset_format="hdf5",
        )
    assert (dataset_dir / "manifest.json").read_text(encoding="utf-8") == "{not json"
