"""Streamed dense persistence: open a dense ArrayStore for a run, then write its manifest.

The arrays are written by SampleWriter into the store opened here, at the same paths the
buffered zarr and hdf5 writers use, so readers see one layout whichever path produced it.
The manifest is written last, from the shapes the writer reports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from neuralls.domain.generation.ports import ArrayStore
from neuralls.domain.generation.sample_writer import (
    MATRIX_NAME,
    MATRIX_SAMPLE_INDEX_NAME,
    RHS_NAME,
    ROW_KIND_NAME,
    SOLUTIONS_NAME,
    DatasetArtifacts,
    StoredArray,
)
from neuralls.platform.sparse_io.protocol import SparseSummary
from neuralls.platform.storage.array_store import Hdf5ArrayStore, ZarrArrayStore
from neuralls.platform.storage.csr_storage import describe_csr_matrix_group
from neuralls.platform.storage.generation_formats import (
    _ZARR_GROUP_NAME,
    HDF5_FILENAME,
    ArtifactLocation,
    ManifestLocations,
    _hdf5_manifest_locations,
    _hdf5_member_location,
    _shape_artifact,
    _zarr_manifest_locations,
    _zarr_member_location,
    build_dense_manifest,
    dense_normalization,
)
from neuralls.platform.storage.manifest import DatasetArtifact
from neuralls.platform.storage.manifest_io import save_dataset_manifest
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX
from neuralls.shared.types import DatasetFormat, LayoutType, MatrixFormat, ScaleMetadata

STREAMED_DENSE_FORMATS: frozenset[DatasetFormat] = frozenset({"zarr", "hdf5"})
"""Dataset formats whose dense arrays can be written batch by batch."""

MemberLocator = Callable[[str], ArtifactLocation]
"""Maps an array name to its manifest address inside the format's container."""


@dataclass(frozen=True)
class StreamedManifestFacts:
    """Run-level values the manifest records that the arrays themselves do not carry.

    Attributes:
        normalization_type: Normalization strategy name.
        matrix_norm: Representative matrix norm of the run.
        matrix_norm_type: Norm type used for ``matrix_norm``.
        scale_metadata: Shared scale payload, or None when bindings disagree.
        layout: Matrix layout the writer chose.
    """

    normalization_type: str
    matrix_norm: float
    matrix_norm_type: str
    scale_metadata: ScaleMetadata | None
    layout: LayoutType


def csr_matrix_container(dataset_dir: Path, dataset_format: DatasetFormat) -> Path:
    """Where a streamed CSR matrix is written: the manifest's matrix address, under ``dataset_dir``.

    zarr stores it as the ``matrix`` group inside ``dataset.zarr``; hdf5 stores it as the
    ``matrix`` group inside ``dataset.h5``, next to the dense arrays.

    Raises:
        ValueError: If the format has no streamed dense backend.
    """
    locations, _ = _manifest_addressing(dataset_format)
    return dataset_dir / locations.matrix.path


def save_csr_stream_manifest(
    dataset_dir: Path,
    dataset_format: DatasetFormat,
    artifacts: DatasetArtifacts,
    facts: StreamedManifestFacts,
    summary: SparseSummary,
) -> None:
    """Write the manifest for a streamed CSR dataset, after its matrix and arrays are complete.

    The non-matrix fields come from the same builder as the dense manifest. The matrix
    descriptor matches the buffered writer of each format: zarr records the descriptor read
    back from the group (``describe_csr_matrix_group``), hdf5 records the builder's own
    descriptor.

    Args:
        dataset_dir: Directory holding the dataset arrays and the CSR matrix.
        dataset_format: ``zarr`` or ``hdf5``.
        artifacts: Dense arrays reported by the writer's ``finalize``; no matrix array.
        facts: Run-level values for the manifest. ``facts.layout`` is not read here; the
            layout comes from ``summary``, which the CSR writer chose.
        summary: Sample count and layout reported by the CSR stream's ``close``.

    Raises:
        ValueError: If a required array (rhs, solutions) is missing from ``artifacts``, or
            the format has no streamed dense backend.
    """
    locations, member = _manifest_addressing(dataset_format)
    rhs = _required(artifacts, RHS_NAME)
    solutions = _required(artifacts, SOLUTIONS_NAME)
    matrix_shape, layout = _csr_matrix_shape_and_layout(
        dataset_dir, dataset_format, locations, summary
    )
    manifest = build_dense_manifest(
        locations=locations,
        matrix_shape=matrix_shape,
        rhs_shape=rhs.shape,
        solutions_shape=solutions.shape,
        normalization=dense_normalization(
            normalization_type=facts.normalization_type,
            matrix_norm=facts.matrix_norm,
            matrix_norm_type=facts.matrix_norm_type,
            scale_metadata=facts.scale_metadata,
        ),
        layout=layout,
        params=_parameter_artifacts(artifacts, dataset_format, member),
        row_kind_shape=_shape_or_none(artifacts, ROW_KIND_NAME),
        matrix_sample_index_shape=_shape_or_none(artifacts, MATRIX_SAMPLE_INDEX_NAME),
        matrix_format=MatrixFormat.CSR,
    )
    save_dataset_manifest(dataset_dir, manifest)


def _csr_matrix_shape_and_layout(
    dataset_dir: Path,
    dataset_format: DatasetFormat,
    locations: ManifestLocations,
    summary: SparseSummary,
) -> tuple[tuple[int, ...], LayoutType]:
    """Shape and layout of the CSR matrix as the buffered writer records them.

    zarr reads them back from the group with ``describe_csr_matrix_group``, which is what the
    buffered zarr writer does. hdf5 uses the writer's own summary.
    """
    if dataset_format != "zarr":
        return (summary.sample_count,), summary.layout
    described = describe_csr_matrix_group(
        csr_matrix_container(dataset_dir, dataset_format), member_path=locations.matrix.path
    )
    if described.layout is None:
        raise ValueError(f"CSR matrix group at '{dataset_dir}' has no recorded layout.")
    return described.shape, described.layout


def open_dense_array_store(dataset_dir: Path, dataset_format: DatasetFormat) -> ArrayStore:
    """Open the dense store for a dataset format at the path its buffered writer uses.

    Args:
        dataset_dir: Directory that will hold the dataset.
        dataset_format: ``zarr`` (group ``dataset.zarr``) or ``hdf5`` (file ``dataset.h5``).

    Returns:
        A store ready to receive arrays.

    Raises:
        ValueError: If the format has no streamed dense backend.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    match dataset_format:
        case "zarr":
            return ZarrArrayStore(dataset_dir / _ZARR_GROUP_NAME)
        case "hdf5":
            return Hdf5ArrayStore(dataset_dir / HDF5_FILENAME)
        case _:
            raise ValueError(f"Streamed dense storage does not support format {dataset_format!r}.")


def save_dense_stream_manifest(
    dataset_dir: Path,
    dataset_format: DatasetFormat,
    artifacts: DatasetArtifacts,
    facts: StreamedManifestFacts,
) -> None:
    """Write the manifest for a streamed dense dataset, after its arrays are complete.

    Args:
        dataset_dir: Directory holding the dataset arrays.
        dataset_format: The format the arrays were written in.
        artifacts: Arrays reported by the writer's ``finalize``.
        facts: Run-level values for the manifest.

    Raises:
        ValueError: If a required array (rhs, solutions, matrix) is missing from ``artifacts``,
            or the format has no streamed dense backend.
    """
    locations, member = _manifest_addressing(dataset_format)
    rhs = _required(artifacts, RHS_NAME)
    solutions = _required(artifacts, SOLUTIONS_NAME)
    matrix = _required(artifacts, MATRIX_NAME)
    manifest = build_dense_manifest(
        locations=locations,
        matrix_shape=matrix.shape,
        rhs_shape=rhs.shape,
        solutions_shape=solutions.shape,
        normalization=dense_normalization(
            normalization_type=facts.normalization_type,
            matrix_norm=facts.matrix_norm,
            matrix_norm_type=facts.matrix_norm_type,
            scale_metadata=facts.scale_metadata,
        ),
        layout=facts.layout,
        params=_parameter_artifacts(artifacts, dataset_format, member),
        row_kind_shape=_shape_or_none(artifacts, ROW_KIND_NAME),
        matrix_sample_index_shape=_shape_or_none(artifacts, MATRIX_SAMPLE_INDEX_NAME),
    )
    save_dataset_manifest(dataset_dir, manifest)


def _manifest_addressing(dataset_format: DatasetFormat) -> tuple[ManifestLocations, MemberLocator]:
    match dataset_format:
        case "zarr":
            return _zarr_manifest_locations(dataset_format), _zarr_member_location
        case "hdf5":
            return _hdf5_manifest_locations(dataset_format), _hdf5_member_location
        case _:
            raise ValueError(f"Streamed dense storage does not support format {dataset_format!r}.")


def _required(artifacts: DatasetArtifacts, name: str) -> StoredArray:
    array = artifacts.get(name)
    if array is None:
        raise ValueError(f"Streamed dataset has no {name!r} array to describe in its manifest.")
    return array


def _shape_or_none(artifacts: DatasetArtifacts, name: str) -> tuple[int, ...] | None:
    array = artifacts.get(name)
    return None if array is None else array.shape


def _parameter_artifacts(
    artifacts: DatasetArtifacts,
    dataset_format: DatasetFormat,
    member: MemberLocator,
) -> tuple[DatasetArtifact, ...]:
    """Parameter descriptors in index order, one per ``parameters_<k>`` array the writer made."""
    indexed: list[tuple[int, StoredArray]] = [
        (int(array.name.removeprefix(PARAMETERS_ZARR_PREFIX)), array)
        for array in artifacts.arrays
        if array.name.startswith(PARAMETERS_ZARR_PREFIX)
    ]
    return tuple(
        _shape_artifact(
            array.shape,
            member(array.name),
            format_name=dataset_format,
            dtype=array.dtype,
            index=index,
        )
        for index, array in sorted(indexed, key=lambda pair: pair[0])
    )


__all__ = [
    "STREAMED_DENSE_FORMATS",
    "StreamedManifestFacts",
    "csr_matrix_container",
    "open_dense_array_store",
    "save_csr_stream_manifest",
    "save_dense_stream_manifest",
]
