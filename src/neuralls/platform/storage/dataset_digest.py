"""Content and stat digests of a generated dataset directory.

`dataset_content_digest` hashes the *logical* arrays (independent of storage
format and location); `stat_digest` is the cheap snapshot that lets
`current_dataset_digest` skip re-hashing when nothing on disk changed.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from neuralls.platform.storage.dataset_readers import (
    DatasetArtifacts,
    ResolvedDatasetArtifact,
    open_resolved_array,
    resolve_dataset_artifacts,
)
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.digest import Digest, array_digest, canonical_digest


def _artifact_roles(artifacts: DatasetArtifacts) -> dict[str, ResolvedDatasetArtifact]:
    """Name every persisted artifact by a stable, format-independent role."""
    roles: dict[str, ResolvedDatasetArtifact] = {
        "matrix": artifacts.matrix,
        "rhs": artifacts.rhs,
        "solutions": artifacts.solutions,
    }
    roles.update({f"params.{i}": param for i, param in enumerate(artifacts.params)})
    if artifacts.row_kind is not None:
        roles["row_kind"] = artifacts.row_kind
    if artifacts.matrix_sample_index is not None:
        roles["matrix_sample_index"] = artifacts.matrix_sample_index
    return roles


def _artifact_digest(artifact: ResolvedDatasetArtifact) -> Digest:
    with open_resolved_array(artifact) as array:
        return array_digest(array)


def dataset_content_digest(data_dir: Path) -> Digest:
    """Recompute the logical-content digest of a dataset from scratch.

    Every manifest artifact is opened lazily through the dataset readers and
    streamed through `array_digest`, so the result is identical for the same
    arrays stored as hdf5, zarr or npy, and does not depend on `data_dir`.

    Args:
        data_dir (Path): Dataset directory holding the manifest.

    Returns:
        Digest: ``sha256:<hex>`` over ``{role: array digest}``.

    Raises:
        FileNotFoundError: If the manifest or an artifact is missing.
    """
    roles = _artifact_roles(resolve_dataset_artifacts(data_dir))
    return canonical_digest({role: _artifact_digest(a) for role, a in roles.items()})


def _reachable_files(path: Path) -> list[Path]:
    """Return the file, or every file under the directory, at ``path``."""
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    if path.is_file():
        return [path]
    raise FileNotFoundError(path)


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def stat_digest(data_dir: Path) -> str:
    """Cheap digest of the manifest artifacts' file stats.

    Covers the sorted ``(path relative to data_dir, size, mtime_ns)`` of every
    file reachable from the manifest (zarr stores are walked) and the
    manifest's own artifact entries. Path independent.

    Args:
        data_dir (Path): Dataset directory holding the manifest.

    Returns:
        str: ``sha256:<hex>`` snapshot digest.

    Raises:
        FileNotFoundError: If the manifest or an artifact is missing.
    """
    roles = _artifact_roles(resolve_dataset_artifacts(data_dir))
    entries: set[tuple[str, int, int]] = set()
    descriptors: dict[str, object] = {}
    for role, artifact in roles.items():
        for file in _reachable_files(artifact.path):
            stat = file.stat()
            entries.add((_relative(file, data_dir), stat.st_size, stat.st_mtime_ns))
        descriptors[role] = {**asdict(artifact), "path": _relative(artifact.path, data_dir)}
    return canonical_digest(sorted(entries), descriptors)


def current_dataset_digest(data_dir: Path) -> Digest:
    """Return the dataset's content digest, re-hashing only when needed.

    O(1) when the manifest stores `content_digest` and `stat_digest` and the
    latter still matches the files on disk (documented blind spot: a same-size
    edit with a restored mtime). Otherwise (legacy manifest, copied or touched
    dataset) the content is recomputed. Never writes.

    Args:
        data_dir (Path): Dataset directory holding the manifest.

    Returns:
        Digest: The dataset's content digest.

    Raises:
        FileNotFoundError: If the manifest or an artifact is missing.
    """
    manifest = read_dataset_manifest(data_dir)
    if (
        manifest.content_digest is not None
        and manifest.stat_digest is not None
        and manifest.stat_digest == stat_digest(data_dir)
    ):
        return manifest.content_digest
    return dataset_content_digest(data_dir)
