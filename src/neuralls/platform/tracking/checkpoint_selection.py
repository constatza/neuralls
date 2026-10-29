"""Shared checkpoint discovery/selection helpers for run-artifact directories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from loguru import logger


def select_preferred_checkpoint(checkpoints: list[Path]) -> tuple[str, Path] | None:
    """Select one preferred checkpoint candidate by explicit filename role."""
    role_candidates: dict[str, list[Path]] = {"best": []}
    for checkpoint in checkpoints:
        match checkpoint.name:
            case "best.ckpt":
                role_candidates["best"].append(checkpoint)
            case _:
                continue

    match role_candidates:
        case {"best": [best_checkpoint]}:
            return "best", best_checkpoint
        case _:
            return None


@dataclass(slots=True, frozen=True)
class CheckpointCandidate:
    """One checkpoint file found under a run root, identified by its content digest."""

    path: Path
    digest: str


def find_single_checkpoint(root: Path) -> Path:
    """Find one unambiguous checkpoint under a local artifact directory."""
    paths = sorted(root.glob("**/*.ckpt"))
    logger.debug(
        "Checkpoint discovery under {} found candidates: {}",
        root,
        [path.as_posix() for path in paths],
    )
    # A lone checkpoint is never ambiguous, so its bytes are not read.
    candidates = [
        CheckpointCandidate(path, sha256(path.read_bytes()).hexdigest() if len(paths) > 1 else "")
        for path in paths
    ]
    return choose_checkpoint(root, candidates)


def choose_checkpoint(root: Path, candidates: Sequence[CheckpointCandidate]) -> Path:
    """Pick the one checkpoint a run root stands for, from candidates already read.

    Identical copies collapse to the shallowest, alphabetically first path. Among distinct
    files, exactly one ``best.ckpt`` wins; anything else is ambiguous.

    Raises:
        FileNotFoundError: There are no candidates.
        ValueError: Several distinct checkpoints remain and none is the single ``best.ckpt``.
    """
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    paths = [candidate.path for candidate in candidates]
    if len(candidates) == 1:
        return paths[0]

    distinct = {(candidate.path.name, candidate.digest) for candidate in candidates}
    if len(distinct) == 1:
        canonical = min(
            paths,
            key=lambda path: (len(path.relative_to(root).parts), path.relative_to(root).as_posix()),
        )
        logger.warning(
            "Duplicate checkpoint artifacts found under {}. Using canonical path {}.",
            root,
            canonical,
        )
        return canonical

    match select_preferred_checkpoint(paths):
        case ("best", best_checkpoint):
            logger.warning(
                "Multiple distinct checkpoint artifacts found under {}. Using best checkpoint {}.",
                root,
                best_checkpoint,
            )
            return best_checkpoint
        case _:
            pass

    candidate_list = ", ".join(path.as_posix() for path in paths)
    raise ValueError(f"Multiple distinct checkpoints found under {root}: {candidate_list}")
