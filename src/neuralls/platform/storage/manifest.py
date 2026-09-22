"""Typed manifest models and serialization for generated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.digest import Digest
from neuralls.shared.types import LayoutType

_DATASET_SCHEMA = "neuralls.dataset.v2"


@dataclass(frozen=True)
class DatasetArtifact:
    """One persisted dataset artifact declared in the manifest."""

    path: str
    format: str
    dtype: str
    shape: tuple[int, ...]
    index: int | None = None
    n_matrix_samples: int | None = None
    broadcast: bool | None = None
    key: str | None = None
    layout: LayoutType | None = None
    logical_sample_count: int | None = None


@dataclass(frozen=True)
class DatasetNormalization:
    """Normalization metadata stored in the dataset manifest."""

    type: str
    matrix_norm: float
    matrix_norm_type: str
    scale: dict[str, Any]


@dataclass(frozen=True)
class DatasetManifest:
    """Typed view over the dataset manifest contract.

    Attributes:
        content_digest: Logical-content digest of every persisted artifact
            (`platform.storage.dataset_digest`), independent of format and
            location. None on legacy manifests.
        stat_digest: Cheap (relative path, size, mtime) snapshot of the
            artifact files taken when `content_digest` was computed; equal
            snapshot means `content_digest` is still valid without re-hashing.
        identity_key: Generation identity key the dataset was produced under.
            A manifest without it is never trusted for reuse.
        identity_components: Short per-input digests behind `identity_key`,
            used to explain why a lookup missed.
    """

    schema: str
    matrix: DatasetArtifact
    rhs: DatasetArtifact
    solutions: DatasetArtifact
    normalization: DatasetNormalization
    params: tuple[DatasetArtifact, ...] = ()
    row_kind: DatasetArtifact | None = None
    matrix_sample_index: DatasetArtifact | None = None
    content_digest: Digest | None = None
    stat_digest: str | None = None
    identity_key: Digest | None = None
    identity_components: dict[str, str] | None = None


def manifest_path_for(dataset_dir: str | Path) -> Path:
    """Return the canonical manifest path for a dataset directory.

    Args:
        dataset_dir: Dataset directory.

    Returns:
        Path to the manifest JSON file.
    """
    return Path(dataset_dir) / DATASET_MANIFEST_FILENAME
