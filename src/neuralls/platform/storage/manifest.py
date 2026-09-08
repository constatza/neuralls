"""Typed manifest models and serialization for generated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
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
        dataset_fingerprint: Fingerprint of the matrix/rhs/solutions artifacts as
            they stood when the dataset was generated, in the same form the
            training reuse-check computes (`platform.caching`). None means the
            manifest predates fingerprinting or was written outside the
            generation pipeline — callers fall back to existence checks alone
            rather than treating the absence as a mismatch.
    """

    schema: str
    matrix: DatasetArtifact
    rhs: DatasetArtifact
    solutions: DatasetArtifact
    normalization: DatasetNormalization
    params: tuple[DatasetArtifact, ...] = ()
    row_kind: DatasetArtifact | None = None
    matrix_sample_index: DatasetArtifact | None = None
    dataset_fingerprint: str | None = None


def manifest_path_for(dataset_dir: str | Path) -> Path:
    """Return the canonical manifest path for a dataset directory.

    Args:
        dataset_dir: Dataset directory.

    Returns:
        Path to the manifest JSON file.
    """
    return Path(dataset_dir) / DATASET_MANIFEST_FILENAME
