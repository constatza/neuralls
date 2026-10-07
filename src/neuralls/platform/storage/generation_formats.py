"""Write-time dataset storage implementations for generation workflows."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.platform.storage.manifest import (
    DatasetArtifact,
    DatasetManifest,
    DatasetNormalization,
)
from neuralls.platform.storage.manifest_io import make_dataset_manifest
from neuralls.shared.types import (
    LayoutType,
    MatrixFormat,
    ScaleMetadata,
)

_HDF5_DEFAULT_CHUNK_ROWS: int = 64


@dataclass(frozen=True)
class ArtifactLocation:
    """Manifest-visible address of one persisted dataset artifact.

    Attributes:
        path: Artifact path recorded in the manifest, relative to the dataset directory.
        key: Optional in-container key, used by container formats such as HDF5.
    """

    path: str
    key: str | None = None


@dataclass(frozen=True)
class ManifestLocations:
    """Per-format manifest addresses for the artifacts every dataset declares."""

    format_name: str
    matrix: ArtifactLocation
    rhs: ArtifactLocation
    solutions: ArtifactLocation
    row_kind: ArtifactLocation
    matrix_sample_index: ArtifactLocation


_CSR_NON_ZARR_MESSAGE: str = (
    "CSR storage is not supported by {format_name} storage. "
    "Use dataset_format='zarr' or dataset_format='hdf5'."
)
_ZARR_MATRIX_STAGING_NAME: str = ".matrix-staging.zarr"
_ZARR_CSR_STAGING_NAME: str = ".matrix-staging-csr.zarr"
_HDF5_CSR_STAGING_NAME: str = ".matrix-staging-csr.h5"


def _shape_artifact(
    shape: tuple[int, ...],
    location: ArtifactLocation,
    *,
    format_name: str,
    dtype: str = "float64",
    index: int | None = None,
) -> DatasetArtifact:
    """Describe one persisted array of the given shape as a manifest artifact entry."""
    return DatasetArtifact(
        path=location.path,
        format=format_name,
        dtype=dtype,
        shape=tuple(int(dim) for dim in shape),
        index=index,
        key=location.key,
    )


def build_dense_manifest(
    *,
    locations: ManifestLocations,
    matrix_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    solutions_shape: tuple[int, ...],
    normalization: DatasetNormalization,
    layout: LayoutType,
    params: tuple[DatasetArtifact, ...],
    row_kind_shape: tuple[int, ...] | None = None,
    matrix_sample_index_shape: tuple[int, ...] | None = None,
    matrix_format: MatrixFormat | None = None,
) -> DatasetManifest:
    """Assemble the dataset manifest from array shapes, shared by every storage writer.

    Buffered writers derive the shapes from a payload and streamed writers from the
    stores they filled. Both produce the same manifest for the same arrays.

    Args:
        locations: Format-specific manifest addresses for each artifact.
        matrix_shape: Physical shape of the persisted matrix artifact.
        rhs_shape: Shape of the RHS artifact, (samples, n).
        solutions_shape: Shape of the solutions artifact, (samples, n).
        normalization: Normalization metadata recorded for the dataset.
        layout: Matrix layout the writer chose.
        params: Parameter artifact descriptors, in manifest order.
        row_kind_shape: Shape of the row-kind artifact, or None when absent.
        matrix_sample_index_shape: Shape of the matrix-index artifact, or None when absent.
        matrix_format: Matrix format to record; None leaves the dense manifest
            unchanged (``manifest_io`` omits a None format).

    Returns:
        Typed dataset manifest ready to be written to disk.
    """
    format_name = locations.format_name
    return make_dataset_manifest(
        matrix=DatasetArtifact(
            path=locations.matrix.path,
            format=format_name,
            dtype="float64",
            shape=matrix_shape,
            n_matrix_samples=int(matrix_shape[0]),
            broadcast=layout == LayoutType.BROADCAST_SINGLE,
            layout=layout,
            logical_sample_count=int(rhs_shape[0]),
            key=locations.matrix.key,
            matrix_format=matrix_format,
        ),
        rhs=_shape_artifact(rhs_shape, locations.rhs, format_name=format_name),
        solutions=_shape_artifact(solutions_shape, locations.solutions, format_name=format_name),
        normalization=normalization,
        params=params,
        row_kind=None
        if row_kind_shape is None
        else _shape_artifact(
            row_kind_shape, locations.row_kind, format_name=format_name, dtype="uint8"
        ),
        matrix_sample_index=None
        if matrix_sample_index_shape is None
        else _shape_artifact(
            matrix_sample_index_shape,
            locations.matrix_sample_index,
            format_name=format_name,
            dtype="int64",
        ),
    )


def dense_normalization(
    *,
    normalization_type: str,
    matrix_norm: float,
    matrix_norm_type: str,
    scale_metadata: ScaleMetadata | None,
) -> DatasetNormalization:
    """Normalization block of the manifest, with an absent scale recorded as an empty mapping."""
    return DatasetNormalization(
        type=normalization_type,
        matrix_norm=float(matrix_norm),
        matrix_norm_type=matrix_norm_type,
        scale=dict(scale_metadata or {}),
    )


_ZARR_GROUP_NAME = "dataset.zarr"


def _zarr_member_location(name: str) -> ArtifactLocation:
    return ArtifactLocation(f"{_ZARR_GROUP_NAME}/{name}")


def _zarr_manifest_locations(format_name: str) -> ManifestLocations:
    return ManifestLocations(
        format_name=format_name,
        matrix=_zarr_member_location("matrix"),
        rhs=_zarr_member_location("rhs"),
        solutions=_zarr_member_location("solutions"),
        row_kind=_zarr_member_location("row_kind"),
        matrix_sample_index=_zarr_member_location("matrix_sample_index"),
    )


HDF5_FILENAME = "dataset.h5"


def _hdf5_member_location(key: str) -> ArtifactLocation:
    return ArtifactLocation(HDF5_FILENAME, key=key)


def _hdf5_manifest_locations(format_name: str) -> ManifestLocations:
    return ManifestLocations(
        format_name=format_name,
        matrix=_hdf5_member_location("matrix"),
        rhs=_hdf5_member_location("rhs"),
        solutions=_hdf5_member_location("solutions"),
        row_kind=_hdf5_member_location("row_kind"),
        matrix_sample_index=_hdf5_member_location("matrix_sample_index"),
    )
