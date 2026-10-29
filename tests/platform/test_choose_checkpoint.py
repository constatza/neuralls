"""`choose_checkpoint` decides on candidates already read, touching no filesystem."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.tracking.checkpoint_selection import CheckpointCandidate, choose_checkpoint

ROOT = Path("run")


def _candidate(relative: str, digest: str) -> CheckpointCandidate:
    return CheckpointCandidate(ROOT / relative, digest)


def test_no_candidate_is_missing() -> None:
    with pytest.raises(FileNotFoundError):
        choose_checkpoint(ROOT, [])


def test_a_lone_candidate_is_chosen() -> None:
    assert choose_checkpoint(ROOT, [_candidate("model.ckpt", "")]) == ROOT / "model.ckpt"


def test_identical_copies_collapse_to_the_shallowest_path() -> None:
    nested = _candidate("checkpoints/model.ckpt", "same")
    top = _candidate("model.ckpt", "same")

    assert choose_checkpoint(ROOT, [nested, top]) == ROOT / "model.ckpt"


def test_the_single_best_checkpoint_beats_the_others() -> None:
    best = _candidate("best.ckpt", "b")

    assert choose_checkpoint(ROOT, [best, _candidate("last.ckpt", "l")]) == best.path


def test_two_distinct_best_checkpoints_are_ambiguous() -> None:
    candidates = [
        _candidate("a/best.ckpt", "a"),
        _candidate("b/best.ckpt", "b"),
        _candidate("last.ckpt", "l"),
    ]

    with pytest.raises(ValueError, match="Multiple distinct checkpoints"):
        choose_checkpoint(ROOT, candidates)


def test_distinct_files_without_a_best_are_ambiguous() -> None:
    candidates = [_candidate("epoch=12.ckpt", "e"), _candidate("last.ckpt", "l")]

    with pytest.raises(ValueError, match="Multiple distinct checkpoints"):
        choose_checkpoint(ROOT, candidates)
