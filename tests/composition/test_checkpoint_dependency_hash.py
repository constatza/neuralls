"""Tests for `_checkpoint_digests`: the checkpoint-content part of a comparison's identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.assignments.comparison_batch import _checkpoint_digests
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A checkpoint file."""
    path = tmp_path / "a" / "model.ckpt"
    path.parent.mkdir()
    path.write_bytes(b"weights-v1")
    return path


@pytest.fixture
def same_bytes_elsewhere(tmp_path: Path) -> Path:
    """Identical bytes at another location (a re-leased temp copy)."""
    path = tmp_path / "b" / "copy.ckpt"
    path.parent.mkdir()
    path.write_bytes(b"weights-v1")
    return path


@pytest.fixture
def retrained(tmp_path: Path) -> Path:
    """A retrained checkpoint with different bytes."""
    path = tmp_path / "c" / "model.ckpt"
    path.parent.mkdir()
    path.write_bytes(b"weights-v2")
    return path


def _resolved(
    path: Path, *, name: str = "a", run_id: str | None = "run-1"
) -> NeuralPreconditionerConfig:
    """A NeuralPreconditionerConfig as it looks after model resolution has run."""
    spec = NeuralPreconditionerConfig(name=name, checkpoint_path=path)
    return spec.model_copy(update={"resolved_checkpoint_path": path, "resolved_run_id": run_id})


def test_same_bytes_at_different_paths_give_the_same_digests(
    checkpoint: Path, same_bytes_elsewhere: Path
) -> None:
    """Identity follows checkpoint content, never the leased temp path."""
    assert _checkpoint_digests([_resolved(checkpoint)]) == _checkpoint_digests(
        [_resolved(same_bytes_elsewhere)]
    )


def test_different_bytes_change_the_digests(checkpoint: Path, retrained: Path) -> None:
    """A retrained model with different weights changes the identity."""
    assert _checkpoint_digests([_resolved(checkpoint)]) != _checkpoint_digests(
        [_resolved(retrained)]
    )


def test_run_id_and_name_do_not_matter(checkpoint: Path) -> None:
    """Only bytes count: a new run id or a rename with identical weights is not a change."""
    assert _checkpoint_digests(
        [_resolved(checkpoint, run_id="r1", name="x")]
    ) == _checkpoint_digests([_resolved(checkpoint, run_id="r2", name="y")])


def test_non_checkpoint_specs_contribute_nothing() -> None:
    """Static preconditioners have no checkpoint dependency."""
    spec = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    assert _checkpoint_digests([spec]) == {}
