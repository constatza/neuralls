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


def test_find_single_checkpoint_prefers_best_over_last(tmp_path: Path) -> None:
    root = tmp_path / "run-download"
    root.mkdir(parents=True)
    best = root / "best.ckpt"
    best.write_bytes(b"best")
    (root / "last.ckpt").write_bytes(b"last")

    resolved = find_single_checkpoint(root)

    assert resolved == best


def test_find_single_checkpoint_rejects_multiple_best_candidates(tmp_path: Path) -> None:
    root = tmp_path / "run-download"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "best.ckpt").write_bytes(b"best-a")
    (root / "b" / "best.ckpt").write_bytes(b"best-b")
    (root / "last.ckpt").write_bytes(b"last")

    with pytest.raises(ValueError):
        find_single_checkpoint(root)


def test_find_single_checkpoint_rejects_interval_checkpoint_with_last(tmp_path: Path) -> None:
    root = tmp_path / "run-download"
    root.mkdir(parents=True)
    (root / "epoch=12-step=500.ckpt").write_bytes(b"interval")
    (root / "last.ckpt").write_bytes(b"last")

    with pytest.raises(ValueError):
        find_single_checkpoint(root)


def test_find_single_checkpoint_rejects_distinct_candidates(tmp_path: Path) -> None:
    """Multiple distinct checkpoint files in one run root should fail loudly."""
    root = tmp_path / "run-download"
    root.mkdir(parents=True)
    (root / "first.ckpt").write_bytes(b"first")
    (root / "second.ckpt").write_bytes(b"second")

    with pytest.raises(ValueError):
        find_single_checkpoint(root)


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
