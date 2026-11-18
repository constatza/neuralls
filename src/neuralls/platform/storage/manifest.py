"""Typed manifest models and serialization for generated datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME, REGENERATE_DATASETS_COMMAND
from neuralls.shared.digest import Digest
from neuralls.shared.system_scaling import SystemScaling
from neuralls.shared.types import LayoutType, MatrixFormat, MatrixSampleId


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """One persisted dataset artifact declared in the manifest."""

    path: str
    format: str
    dtype: str
    shape: tuple[int, ...]
    index: int | None = None
    n_matrix_samples: int | None = None
    key: str | None = None
    layout: LayoutType | None = None
    logical_sample_count: int | None = None
    matrix_format: MatrixFormat | None = None
    """Matrix storage format; None, when the manifest omits it, reads as dense."""


@dataclass(frozen=True, slots=True)
class DatasetScaling:
    """How a dataset's systems were scaled, stored in the manifest under ``scaling``.

    Attributes:
        scaling: The scaling every matrix of the dataset was generated with.
        matrix_norm: Norm of the first scaled matrix (representative).
        matrix_norm_type: Norm type of ``matrix_norm``.
        matrix_scales: Scale factor ``alpha`` of each matrix sample id; the stored matrix
            is the raw matrix divided by it.
    """

    scaling: SystemScaling
    matrix_norm: float
    matrix_norm_type: str
    matrix_scales: Mapping[MatrixSampleId, float]

    def for_matrix(self, matrix_id: MatrixSampleId) -> PersistedMatrixScaling:
        """The scaling of one stored matrix, by its matrix sample id.

        Raises:
            KeyError: If the dataset recorded no scale factor for ``matrix_id``.
        """
        alpha = self.matrix_scales.get(matrix_id)
        if alpha is None:
            raise KeyError(
                f"The dataset manifest records no scale factor for matrix {matrix_id}; "
                f"it has factors for {sorted(self.matrix_scales)}."
            )
        return PersistedMatrixScaling(scaling=self.scaling, alpha=alpha)


@dataclass(frozen=True, slots=True)
class PersistedMatrixScaling:
    """The scaling one stored matrix carries: the dataset's scaling and this matrix's factor.

    Attributes:
        scaling: The scaling the dataset was generated with.
        alpha: The factor this matrix was divided by; a raw RHS for it is divided by the same.
    """

    scaling: SystemScaling
    alpha: float


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Typed view over the dataset manifest contract.

    Attributes:
        content_digest: Logical-content digest of every persisted artifact
            (`platform.storage.dataset_digest`), independent of format and
            location. None when the manifest carries no digest.
        stat_digest: Cheap (relative path, size, mtime) snapshot of the
            artifact files taken when `content_digest` was computed; equal
            snapshot means `content_digest` is still valid without re-hashing.
        identity_key: Generation identity key the dataset was produced under.
            A manifest without it is never trusted for reuse.
        identity_components: Short per-input digests behind `identity_key`,
            used to explain why a lookup missed.
        generation_duration_seconds: Wall time the generation that produced
            this dataset took, stamped once alongside the content/stat
            digests. `None` on manifests predating this field, or when
            generation was skipped (reused an already-valid dataset) before
            this field existed — a skipped generation after this field
            exists instead leaves whatever was already stamped untouched,
            since the skip path never rewrites the manifest.
        generation_peak_memory_bytes: Peak memory during that generation,
            alongside `generation_duration_seconds`.
        provenance: Structured scientific description of the dataset (its source
            problem and every generation strategy), stamped after writing like the
            digests. None means the dataset has no scientific provenance and must be
            regenerated before it can be trained on, compared, or labelled.
    """

    matrix: DatasetArtifact
    rhs: DatasetArtifact
    solutions: DatasetArtifact
    scaling: DatasetScaling
    params: tuple[DatasetArtifact, ...] = ()
    row_kind: DatasetArtifact | None = None
    matrix_sample_index: DatasetArtifact | None = None
    content_digest: Digest | None = None
    stat_digest: str | None = None
    identity_key: Digest | None = None
    identity_components: dict[str, str] | None = None
    generation_duration_seconds: float | None = None
    generation_peak_memory_bytes: int | None = None
    provenance: DatasetPresentation | None = None


class MissingDatasetProvenanceError(ValueError):
    """A dataset has no scientific provenance, so it cannot be described."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        super().__init__(
            f"Dataset at '{dataset_dir}' has no scientific provenance. Regenerate it with "
            f"`{REGENERATE_DATASETS_COMMAND}` (the dataset config needs a '[source] problem' "
            "name) so its source and generation parameters are recorded."
        )


class InvalidDatasetManifestError(ValueError):
    """A dataset manifest block is missing or malformed, e.g. a manifest from before a format change.

    Attributes:
        dataset_dir: The dataset whose manifest could not be read.
        field: The top-level manifest key whose block could not be decoded.
    """

    def __init__(self, dataset_dir: str | Path, field: str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.field = field
        super().__init__(
            f"Dataset manifest at '{dataset_dir}' has a missing or malformed '{field}' block "
            f"(for example written by an older version). Regenerate it with "
            f"`{REGENERATE_DATASETS_COMMAND}`."
        )


def manifest_path_for(dataset_dir: str | Path) -> Path:
    """Return the canonical manifest path for a dataset directory.

    Args:
        dataset_dir: Dataset directory.

    Returns:
        Path to the manifest JSON file.
    """
    return Path(dataset_dir) / DATASET_MANIFEST_FILENAME
