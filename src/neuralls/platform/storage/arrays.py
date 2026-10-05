"""System data loading utilities.

This module handles loading matrices, RHS vectors, and complete linear systems
from various file formats. All functions are I/O actions with clear contracts.

Functions in this module perform I/O operations and are not pure.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from neuralls.platform.storage.datasets import (
    load_dataset_manifest,
    load_dense_training_arrays,
    load_matrix_dense_sample,
)
from neuralls.platform.storage.matrix_readers import read_matrix, to_dense
from neuralls.shared.types import SystemMatrix


def _load_matrix_file(matrix_path: str | Path) -> SystemMatrix:
    """Load matrix from dataset directory or file.

    I/O action - reads matrix from disk.

    Args:
        matrix_path: Path to dataset directory or matrix file

    Returns:
        Dense ndarray or CSR array, depending on the file format
    """
    matrix_path = Path(matrix_path)

    # If it's a dataset directory, load first matrix sample.
    if matrix_path.is_dir():
        load_dataset_manifest(matrix_path)
        return load_matrix_dense_sample(matrix_path, sample_index=0)

    return read_matrix(matrix_path)


def _load_rhs_file(rhs_path: str | Path) -> np.ndarray:
    """Load RHS vector from dataset directory or file.

    I/O action - reads RHS from disk.

    Args:
        rhs_path: Path to dataset directory or RHS file

    Returns:
        RHS vector as float64, reshaped to 1D
    """
    rhs_path = Path(rhs_path)

    def _validate_and_flatten(rhs_array: np.ndarray) -> np.ndarray:
        if rhs_array.ndim == 1:
            return rhs_array
        if rhs_array.ndim == 2:
            if rhs_array.shape[1] == 1 or rhs_array.shape[0] == 1:
                return rhs_array.reshape(-1)
            raise ValueError(
                f"RHS array must be 1D or a single column/row vector, got shape {rhs_array.shape}"
            )
        raise ValueError(f"RHS array must be 1D or 2D, got shape {rhs_array.shape}")

    # If it's a dataset directory, load first sample from rhs.npy.
    if rhs_path.is_dir():
        try:
            rhs, _ = load_dense_training_arrays(rhs_path)
            b = rhs[0]
            return _validate_and_flatten(b)
        except FileNotFoundError, ValueError:
            pass

    if rhs_path.name == "rhs.npy" and rhs_path.exists():
        b = np.load(rhs_path).astype(np.float64, copy=False)
        if b.ndim > 1:
            b = b[0]
        return _validate_and_flatten(b)

    # Handle .npy binary format or text files
    if rhs_path.suffix == ".npy":
        b = np.load(rhs_path).astype(np.float64, copy=False)
    else:
        b = np.loadtxt(rhs_path, dtype=np.float64)
    return _validate_and_flatten(b)


def _ensure_rhs_matches_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ensure RHS dimension matches matrix, creating fallback if needed.

    Pure function - validates and potentially creates fallback.

    Args:
        A: System matrix
        b: RHS vector

    Returns:
        RHS vector with correct dimension
    """
    if b.shape[0] != A.shape[0]:
        hint = ""
        if b.size == A.size:
            hint = " (loaded data looks like a flattened matrix; check your RHS path)"
        raise ValueError(f"RHS length {b.shape[0]} doesn't match matrix size {A.shape[0]}{hint}")
    return b


def load_system_data(
    matrix_path: str | Path, rhs_path: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load matrix and RHS data.

    Orchestrates loading and validation of system components.

    I/O orchestrator - coordinates file loading.

    Args:
        matrix_path: Path to matrix file
        rhs_path: Path to RHS file (optional)

    Returns:
        Tuple of (matrix, rhs) arrays
    """
    A = to_dense(_load_matrix_file(matrix_path))

    rhs_path = Path(rhs_path) if rhs_path is not None else None
    if rhs_path is None:
        return A, None

    b = _load_rhs_file(rhs_path)
    b = _ensure_rhs_matches_matrix(A, b)

    return A, b


def save_training_data(
    features: np.ndarray,
    targets: np.ndarray,
    features_path: str | Path,
    targets_path: str | Path,
) -> None:
    """Save training data arrays.

    I/O action - writes arrays to disk.

    Args:
        features: Feature array
        targets: Target array
        features_path: Output path for features
        targets_path: Output path for targets
    """
    from neuralls.platform.storage.filesystem import ensure_dir

    features_path = Path(features_path)
    targets_path = Path(targets_path)

    ensure_dir(features_path.parent)
    ensure_dir(targets_path.parent)

    # Ensure float64 dtype for all saved data
    np.save(features_path, features.astype(np.float64, copy=False))
    np.save(targets_path, targets.astype(np.float64, copy=False))


def load_case_data(
    data_dir: str | Path,
    variant: str = "dataset",
) -> Mapping[str, np.ndarray | dict[str, Any]]:
    """Load all data from a case directory.

    I/O action - loads complete dataset from directory.

    Args:
        data_dir: Path to case directory (e.g., collect-504-norm)
        variant: Dataset variant to load ('dataset')

    Returns:
        Dictionary with keys: matrix, rhs, solutions, metadata

    Example:
        >>> data = load_case_data("/data/projects/graph-cg/data/processed/collect-504-norm")
        >>> A = data["matrix"]
        >>> R = data["rhs"]
        >>> X = data["solutions"]
    """
    from neuralls.platform.storage.dataset_views import load_dataset

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load dataset using split sparse format
    result = load_dataset(data_dir, variant=variant)

    # Load metadata
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            result["metadata"] = json.load(f)

    return result
