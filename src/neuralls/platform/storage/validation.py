"""Storage validation utilities for the platform layer."""

from __future__ import annotations

from pathlib import Path

import zarr
from zarr.errors import GroupNotFoundError

from neuralls.platform.storage.csr_layout import (
    DATA_ARRAY,
    INDICES_ARRAY,
    INDPTR_ARRAY,
    SAMPLE_OFFSETS_ARRAY,
    SHAPE_ARRAY,
)
from neuralls.platform.storage.datasets import load_dataset_manifest, resolve_dataset_artifacts
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME

_SUPPORTED_COMPARISON_FILE_SUFFIXES = {"", ".npy", ".txt"}
_CSR_MATRIX_GROUP_MEMBERS = frozenset(
    {INDPTR_ARRAY, INDICES_ARRAY, DATA_ARRAY, SAMPLE_OFFSETS_ARRAY, SHAPE_ARRAY}
)


def _is_csr_matrix_group(path: Path) -> bool:
    """Return True when ``path`` is a zarr group holding every CSR storage member."""
    try:
        group = zarr.open_group(str(path), mode="r")
    except FileNotFoundError, ValueError, GroupNotFoundError:
        return False
    return _CSR_MATRIX_GROUP_MEMBERS.issubset(group.array_keys())


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
                "directory (zarr group with indptr, indices, data, sample_offsets and shape)."
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
