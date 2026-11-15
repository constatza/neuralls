"""Checkpoint discovery in a run's downloaded artifact directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.tracking.checkpoint_selection import find_single_checkpoint


def test_find_single_checkpoint_collapses_nested_duplicate_copy(tmp_path: Path) -> None:
    """Nested duplicate artifact copies should resolve to one canonical checkpoint."""
    root = tmp_path / "run-download"
    duplicate_root = root / "checkpoints"
    root.mkdir(parents=True)
    duplicate_root.mkdir(parents=True)
    primary = root / "model.ckpt"
    duplicate = duplicate_root / "model.ckpt"
    primary.write_bytes(b"same-checkpoint")
    duplicate.write_bytes(b"same-checkpoint")

    resolved = find_single_checkpoint(root)

    assert resolved == primary


def test_find_single_checkpoint_stays_within_provided_root(tmp_path: Path) -> None:
    """Checkpoint discovery must not inspect sibling download roots."""
    target_root = tmp_path / "run-a"
    sibling_root = tmp_path / "run-b"
    target_root.mkdir(parents=True)
    sibling_root.mkdir(parents=True)
    expected = target_root / "model.ckpt"
    expected.write_bytes(b"target")
    (sibling_root / "other.ckpt").write_bytes(b"sibling")

    resolved = find_single_checkpoint(target_root)

    assert resolved == expected


def test_find_single_checkpoint_of_an_empty_root_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_single_checkpoint(tmp_path)
