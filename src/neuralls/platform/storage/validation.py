"""Storage validation utilities for the platform layer."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from neuralls.platform.sparse_io.protocol import SparseLocation
from neuralls.platform.sparse_io.registry import backend_for
from neuralls.platform.storage.datasets import load_dataset_manifest, resolve_dataset_artifacts
from neuralls.platform.storage.generation_formats import HDF5_FILENAME
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.types import DatasetFormat

_SUPPORTED_COMPARISON_FILE_SUFFIXES = {"", ".npy", ".txt"}
_ZARR_FORMAT: Final[DatasetFormat] = "zarr"
_HDF5_FORMAT: Final[DatasetFormat] = "hdf5"
_PROBE_SAMPLE_INDEX: Final = 0
_UNREADABLE_CSR_ERRORS: Final = (KeyError, IndexError, OSError, TypeError, ValueError)


def _csr_probe_target(path: Path) -> tuple[DatasetFormat, SparseLocation]:
    """Pick the storage format and location that a bare CSR directory would hold.

    A dataset directory written by the hdf5 generation path keeps its matrix
    inside ``HDF5_FILENAME``; anything else is probed as a zarr group rooted at
    ``path`` itself. The filename is the only storage hint available here
    because this runs only when the manifest did not load, so there is no
    manifest artifact to consult.
    """
    hdf5_file = path / HDF5_FILENAME
    if hdf5_file.is_file():
        return _HDF5_FORMAT, SparseLocation(path=hdf5_file)
    return _ZARR_FORMAT, SparseLocation(path=path)


def _is_csr_matrix_group(path: Path) -> bool:
    """Return True when ``path`` holds a CSR batch that the registered reader can load.

    The probe is backend-neutral so that every storage format registered in
    ``sparse_io`` is accepted by the same rule. Reading one sample exercises
    every member the reader needs for that layout, so a missing member fails
    inside the reader with its own error (a missing key or group, a missing
    file), and this function only translates that failure into ``False``.
    Asking the reader rather than listing member names here keeps the
    validation from drifting from the schema each backend actually reads.
    """
    format_name, location = _csr_probe_target(path)
    reader, _ = backend_for(format_name)
    try:
        reader.read_sample(location, _PROBE_SAMPLE_INDEX)
    except _UNREADABLE_CSR_ERRORS:
        return False
    return True


def validate_data_exists(
    data_dir: Path | str,
    required_files: list[str],
) -> None:
    """Validate that required data files exist in a directory.

    Args:
        data_dir: Directory to check for files.
        required_files: List of filenames that must exist (e.g.,
            ["rhs-samples.npy", "sol-samples.npy"]).

    Raises:
        FileNotFoundError: If any required file is missing, with a descriptive
            error message listing all missing file paths.
    """
    data_dir = Path(data_dir)
    missing_files = [
        str(data_dir / filename)
        for filename in required_files
        if not (data_dir / filename).exists()
    ]
    if missing_files:
        files_str = "\n  - ".join(missing_files)
        raise FileNotFoundError(f"Required data files not found in {data_dir}:\n  - {files_str}")


def build_missing_input_error(path: Path) -> FileNotFoundError:
    """Build a user-facing missing-input error for a comparison dataset path."""
    return FileNotFoundError(f"Comparison input not found: {path}.")


def validate_comparison_matrix_input(path: Path) -> None:
    """Validate one comparison matrix input without executing the workflow."""
    if not path.exists():
        raise build_missing_input_error(path)
    if not path.is_dir():
        if path.suffix not in _SUPPORTED_COMPARISON_FILE_SUFFIXES:
            raise ValueError(
                f"Unsupported comparison matrix input format: {path}. "
                "Use a dataset directory, .npy file, or text matrix file."
            )
        return
    try:
        load_dataset_manifest(path)
    except FileNotFoundError, ValueError:
        if not _is_csr_matrix_group(path):
            raise ValueError(
                f"Comparison matrix dataset directory is not loadable: {path}. "
                f"Expected a dataset root with {DATASET_MANIFEST_FILENAME} or a CSR matrix "
                "directory (a zarr group or an hdf5 file holding the CSR matrix group)."
            ) from None
        return
    matrix_artifact = resolve_dataset_artifacts(path).matrix
    if not matrix_artifact.path.exists():
        raise ValueError(
            f"Comparison matrix dataset directory is missing {matrix_artifact.path.name}: {path}"
        )


def validate_comparison_rhs_input(path: Path) -> None:
    """Validate one comparison RHS input without executing the workflow."""
    if not path.exists():
        raise build_missing_input_error(path)
    if not path.is_dir():
        if path.suffix not in _SUPPORTED_COMPARISON_FILE_SUFFIXES:
            raise ValueError(
                f"Unsupported comparison RHS input format: {path}. "
                "Use a dataset directory, .npy file, or text vector file."
            )
        return
    load_dataset_manifest(path)
    rhs_artifact = resolve_dataset_artifacts(path).rhs
    if not rhs_artifact.path.exists():
        raise ValueError(
            f"Comparison RHS dataset directory is missing {rhs_artifact.path.name}: {path}"
        )


def validate_comparison_inputs(
    *,
    matrix_path: Path,
    rhs_path: Path,
) -> None:
    """Validate comparison matrix/RHS inputs before opening tracking runs."""
    validate_comparison_matrix_input(matrix_path)
    validate_comparison_rhs_input(rhs_path)
