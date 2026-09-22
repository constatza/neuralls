"""Manifest-driven dataset readers and artifact resolution helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import zarr

from neuralls.domain.solver.utils.validation import validate_ax_equals_b
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.digest import RowSliceable
from neuralls.shared.enum_codecs import decode_row_kind_array
from neuralls.shared.types import LayoutType, RowKind


@dataclass(frozen=True)
class ResolvedDatasetArtifact:
    """Resolved on-disk artifact contract for one dataset array."""

    path: Path
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
class DatasetArtifacts:
    """Explicit resolved dataset contract used by storage readers."""

    root: Path
    manifest_path: Path
    matrix: ResolvedDatasetArtifact
    rhs: ResolvedDatasetArtifact
    solutions: ResolvedDatasetArtifact
    normalization: DatasetNormalization
    params: tuple[ResolvedDatasetArtifact, ...] = ()
    row_kind: ResolvedDatasetArtifact | None = None
    matrix_sample_index: ResolvedDatasetArtifact | None = None


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved dataset artifact paths from the manifest."""

    root: Path
    manifest_path: Path
    rhs_path: Path
    solutions_path: Path
    matrix_path: Path
    parameter_paths: tuple[Path, ...]


@dataclass(frozen=True)
class CanonicalTrainingTriplet:
    """One manifest-backed `(A, b, x)` dataset row with provenance."""

    matrix: np.ndarray
    rhs: np.ndarray
    lhs: np.ndarray
    sample_index: int
    matrix_index: int
    row_kind: RowKind | None
    matrix_binding_enforced: bool
    normalization: DatasetNormalization


def _resolve_artifact(root: Path, artifact: DatasetArtifact) -> ResolvedDatasetArtifact:
    return ResolvedDatasetArtifact(
        path=root / artifact.path,
        format=artifact.format,
        dtype=artifact.dtype,
        shape=artifact.shape,
        index=artifact.index,
        n_matrix_samples=artifact.n_matrix_samples,
        broadcast=artifact.broadcast,
        key=artifact.key,
        layout=artifact.layout,
        logical_sample_count=artifact.logical_sample_count,
    )


def resolve_dataset_artifacts(dataset_dir: str | Path) -> DatasetArtifacts:
    """Resolve the explicit dataset artifact contract from the manifest."""
    root = Path(dataset_dir)
    try:
        manifest = read_dataset_manifest(root)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset manifest not found in '{root}'. "
            "Re-run the generation pipeline to produce a valid dataset."
        ) from None
    return DatasetArtifacts(
        root=root,
        manifest_path=root / DATASET_MANIFEST_FILENAME,
        matrix=_resolve_artifact(root, manifest.matrix),
        rhs=_resolve_artifact(root, manifest.rhs),
        solutions=_resolve_artifact(root, manifest.solutions),
        normalization=manifest.normalization,
        params=tuple(_resolve_artifact(root, artifact) for artifact in manifest.params),
        row_kind=_resolve_artifact(root, manifest.row_kind)
        if manifest.row_kind is not None
        else None,
        matrix_sample_index=(
            _resolve_artifact(root, manifest.matrix_sample_index)
            if manifest.matrix_sample_index is not None
            else None
        ),
    )


def resolve_dataset_paths(dataset_dir: str | Path) -> DatasetPaths:
    """Resolve dataset artifact paths from the manifest contract."""
    artifacts = resolve_dataset_artifacts(dataset_dir)
    return DatasetPaths(
        root=artifacts.root,
        manifest_path=artifacts.manifest_path,
        rhs_path=artifacts.rhs.path,
        solutions_path=artifacts.solutions.path,
        matrix_path=artifacts.matrix.path,
        parameter_paths=tuple(artifact.path for artifact in artifacts.params),
    )


def _load_sample(artifact: ResolvedDatasetArtifact, sample_index: int) -> np.ndarray:
    match artifact.format:
        case "npy":
            arr = np.load(artifact.path, mmap_mode="r")
            return np.asarray(arr[sample_index] if arr.ndim > 1 else arr, dtype=np.float64)
        case "zarr":
            arr = zarr.open_array(str(artifact.path), mode="r")
            data = arr[sample_index] if arr.ndim > 1 else arr[:]
            return np.asarray(data, dtype=np.float64)
        case "hdf5":
            if artifact.key is None:
                raise ValueError(f"HDF5 artifact at {artifact.path} is missing a required key")
            with h5py.File(str(artifact.path), "r") as f:
                ds = f[artifact.key]
                data = ds[sample_index] if ds.ndim > 1 else ds[:]  # type: ignore[index]
                return np.asarray(data, dtype=np.float64)
        case _:
            raise ValueError(f"Unsupported format {artifact.format!r} at {artifact.path}")


def _select_canonical_row(
    *,
    row_kinds: tuple[RowKind, ...] | None,
    sample_index: int | None,
    sample_count: int,
    require_standard: bool,
    dataset_dir: str | Path,
) -> tuple[int, RowKind | None]:
    """Resolve one row index under the active semantic row policy."""
    if sample_index is not None:
        if sample_index < 0 or sample_index >= sample_count:
            raise IndexError(
                f"sample_index={sample_index} out of range for {dataset_dir} "
                f"(available rows: 0..{sample_count - 1})."
            )
        row_kind = row_kinds[sample_index] if row_kinds is not None else None
        if require_standard and row_kind is RowKind.CG_INTERNAL:
            raise ValueError(
                f"sample_index={sample_index} is CG_INTERNAL and cannot be used "
                "with standard-row enforcement."
            )
        return sample_index, row_kind

    if row_kinds is None:
        if require_standard:
            raise ValueError(
                f"Dataset '{dataset_dir}' does not expose row_kind metadata; "
                "cannot select a canonical non-residual row."
            )
        return 0, None

    for index, row_kind in enumerate(row_kinds):
        if not require_standard or row_kind is RowKind.STANDARD:
            return index, row_kind
    raise ValueError(f"No canonical non-residual rows are available in dataset '{dataset_dir}'.")


def _resolve_triplet_matrix_index(
    *,
    artifacts: DatasetArtifacts,
    row_index: int,
    matrix_index: int | None,
    dataset_dir: str | Path,
) -> tuple[int, bool]:
    """Resolve the physical matrix index for a logical dataset row."""
    if artifacts.matrix_sample_index is None:
        if matrix_index is None:
            raise ValueError(
                f"Dataset '{dataset_dir}' does not expose matrix_sample_index metadata; "
                "provide an explicit matrix_index only for legacy single-matrix datasets."
            )
        return matrix_index, False

    bindings = _load_resolved_native(artifacts.matrix_sample_index).astype(np.int64, copy=False)
    bindings = bindings.reshape(-1)
    if row_index >= bindings.shape[0]:
        raise IndexError(
            f"sample_index={row_index} has no matrix_sample_index binding in '{dataset_dir}'."
        )
    return int(bindings[row_index]), True


def resolve_canonical_training_triplet(
    dataset_dir: str | Path,
    sample_index: int | None,
    *,
    require_standard: bool = True,
    matrix_index: int | None = None,
) -> CanonicalTrainingTriplet:
    """Load one canonical manifest-backed `(matrix, rhs, lhs)` dataset row.

    When `sample_index` is omitted, the first STANDARD row is selected. Matrix
    selection follows persisted `matrix_sample_index` bindings; explicit
    `matrix_index` is only a fallback for datasets that do not expose bindings.
    """
    artifacts = resolve_dataset_artifacts(dataset_dir)
    sample_count = read_training_sample_count(artifacts)
    row_kind_codes = load_optional_row_kind_codes(dataset_dir)
    row_kinds = decode_row_kind_array(row_kind_codes) if row_kind_codes is not None else None
    row_index, row_kind = _select_canonical_row(
        row_kinds=row_kinds,
        sample_index=sample_index,
        sample_count=sample_count,
        require_standard=require_standard,
        dataset_dir=dataset_dir,
    )
    resolved_matrix_index, matrix_binding_enforced = _resolve_triplet_matrix_index(
        artifacts=artifacts,
        row_index=row_index,
        matrix_index=matrix_index,
        dataset_dir=dataset_dir,
    )
    matrix = _load_sample(artifacts.matrix, resolved_matrix_index)
    rhs = _load_sample(artifacts.rhs, row_index).reshape(-1)
    lhs = _load_sample(artifacts.solutions, row_index).reshape(-1)
    # STANDARD rows are genuine Ax=b pairs; CG_INTERNAL rows are CG
    # residual/correction pairs with no such relationship — only validate
    # the former. Checked here, on the raw dataset, before any comparison
    # logic runs, so generation-time corruption is caught at its source.
    if row_kind is None or row_kind is RowKind.STANDARD:
        validate_ax_equals_b(matrix, rhs, lhs)
    return CanonicalTrainingTriplet(
        matrix=matrix,
        rhs=rhs,
        lhs=lhs,
        sample_index=row_index,
        matrix_index=resolved_matrix_index,
        row_kind=row_kind,
        matrix_binding_enforced=matrix_binding_enforced,
        normalization=artifacts.normalization,
    )


def try_read_dataset_normalization(path: str | Path) -> DatasetNormalization | None:
    """Return persisted normalization metadata iff `path` is a manifest-backed dataset dir.

    Args:
        path: Candidate dataset directory or raw file path.

    Returns:
        The dataset's persisted `DatasetNormalization`, or `None` when `path`
        is not a directory (i.e. a raw/external file with no manifest).
    """
    candidate = Path(path)
    if not candidate.is_dir():
        return None
    return resolve_dataset_artifacts(candidate).normalization


def _load_resolved(artifact: ResolvedDatasetArtifact) -> np.ndarray:
    match artifact.format:
        case "zarr":
            return np.asarray(zarr.open_array(str(artifact.path), mode="r"), dtype=np.float64)
        case "npy":
            return np.load(artifact.path).astype(np.float64, copy=False)
        case "hdf5":
            if artifact.key is None:
                raise ValueError(f"HDF5 artifact at {artifact.path} is missing a required key")
            with h5py.File(str(artifact.path), "r") as f:
                return np.asarray(f[artifact.key], dtype=np.float64)
        case _:
            raise ValueError(
                f"Unsupported dataset artifact format {artifact.format!r} at {artifact.path}"
            )


@contextmanager
def open_resolved_array(artifact: ResolvedDatasetArtifact) -> Iterator[RowSliceable]:
    """Open one dataset artifact lazily, without coercing dtype or loading it.

    The yielded object supports axis-0 slicing, so callers can stream it in
    chunks. It is only valid inside the ``with`` block (HDF5 closes its file).

    Args:
        artifact (ResolvedDatasetArtifact): Artifact to open.

    Yields:
        RowSliceable: Lazy (memmap / zarr / h5py) view of the stored array.

    Raises:
        ValueError: If the format is unsupported or an HDF5 key is missing.
    """
    match artifact.format:
        case "zarr":
            yield zarr.open_array(str(artifact.path), mode="r")  # ty: ignore[invalid-yield]
        case "npy":
            yield np.load(artifact.path, mmap_mode="r")
        case "hdf5":
            if artifact.key is None:
                raise ValueError(f"HDF5 artifact at {artifact.path} is missing a required key")
            with h5py.File(str(artifact.path), "r") as f:
                yield f[artifact.key]
        case _:
            raise ValueError(
                f"Unsupported dataset artifact format {artifact.format!r} at {artifact.path}"
            )


def _load_resolved_native(artifact: ResolvedDatasetArtifact) -> np.ndarray:
    """Load one dataset artifact without coercing the dtype."""
    match artifact.format:
        case "zarr":
            return np.asarray(zarr.open_array(str(artifact.path), mode="r"))
        case "npy":
            return np.load(artifact.path)
        case "hdf5":
            if artifact.key is None:
                raise ValueError(f"HDF5 artifact at {artifact.path} is missing a required key")
            with h5py.File(str(artifact.path), "r") as f:
                return np.asarray(f[artifact.key])
        case _:
            raise ValueError(
                f"Unsupported dataset artifact format {artifact.format!r} at {artifact.path}"
            )


def load_dense_training_arrays(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load dense rhs and solution arrays from a manifest-driven dataset directory."""
    artifacts = resolve_dataset_artifacts(dataset_dir)
    return _load_resolved(artifacts.rhs), _load_resolved(artifacts.solutions)


@lru_cache(maxsize=128)
def load_matrix_dense_sample(dataset_dir: str | Path, sample_index: int = 0) -> np.ndarray:
    """Load one dense matrix sample from a manifest-driven dataset directory."""
    return _load_sample(resolve_dataset_artifacts(dataset_dir).matrix, sample_index)


def read_training_sample_count(paths: DatasetPaths | DatasetArtifacts) -> int:
    """Return the number of training samples without assuming a fixed layout."""
    if isinstance(paths, DatasetArtifacts):
        rhs_n = int(paths.rhs.shape[0])
        sol_n = int(paths.solutions.shape[0])
    else:
        artifacts = resolve_dataset_artifacts(paths.root)
        rhs_n = int(artifacts.rhs.shape[0])
        sol_n = int(artifacts.solutions.shape[0])
    if rhs_n != sol_n:
        raise ValueError(f"RHS and solutions sample counts must match, got {rhs_n} and {sol_n}")
    return rhs_n


def load_parameter_arrays(dataset_dir: str | Path) -> tuple[np.ndarray, ...]:
    """Load all manifest-declared parameter arrays eagerly."""
    artifacts = resolve_dataset_artifacts(dataset_dir)
    return tuple(_load_resolved(a) for a in artifacts.params)


def load_row_kind_codes(dataset_dir: str | Path) -> np.ndarray:
    """Load persisted compact row-kind semantic codes for a dataset."""
    artifact = resolve_dataset_artifacts(dataset_dir).row_kind
    if artifact is None:
        raise ValueError(
            f"Dataset '{dataset_dir}' does not expose row_kind metadata. Regenerate it first."
        )
    return _load_resolved_native(artifact).astype(np.uint8, copy=False).reshape(-1)


def load_optional_row_kind_codes(dataset_dir: str | Path) -> np.ndarray | None:
    """Load row-kind semantic codes when present, otherwise return None."""
    artifact = resolve_dataset_artifacts(dataset_dir).row_kind
    if artifact is None:
        return None
    return _load_resolved_native(artifact).astype(np.uint8, copy=False).reshape(-1)


def load_matrix_sample_index(dataset_dir: str | Path) -> np.ndarray:
    """Load persisted matrix-sample bindings for a dataset."""
    artifact = resolve_dataset_artifacts(dataset_dir).matrix_sample_index
    if artifact is None:
        raise ValueError(
            f"Dataset '{dataset_dir}' does not expose matrix_sample_index metadata. Regenerate it first."
        )
    return _load_resolved_native(artifact).astype(np.int64, copy=False).reshape(-1)


def list_available_matrix_indices(dataset_dir: str | Path) -> tuple[int, ...]:
    """List valid physical matrix indices for a dataset."""
    matrix = resolve_dataset_artifacts(dataset_dir).matrix
    if matrix.n_matrix_samples is not None:
        return tuple(range(int(matrix.n_matrix_samples)))
    if matrix.shape:
        return tuple(range(int(matrix.shape[0])))
    return ()


def load_standard_row_indices(
    dataset_dir: str | Path, require_standard: bool = True
) -> tuple[int, ...]:
    """List row indices that are STANDARD (safe for solver comparison)."""
    from neuralls.shared.enum_codecs import decode_row_kind_array
    from neuralls.shared.types import RowKind

    codes = load_row_kind_codes(dataset_dir)
    kinds = decode_row_kind_array(codes)
    if not require_standard:
        return tuple(range(len(kinds)))
    return tuple(index for index, kind in enumerate(kinds) if kind is RowKind.STANDARD)
