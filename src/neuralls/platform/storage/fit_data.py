"""Filesystem adapter that reads fit arrays from a manifest-driven dataset directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array

from neuralls.platform.storage.dataset_readers import (
    load_dense_training_arrays,
    load_matrix_sample_index,
    load_matrix_sparse_sample,
    load_parameter_arrays,
    matrix_row,
    resolve_dataset_artifacts,
)


@dataclass(frozen=True, slots=True)
class FsFitDataLoader:
    """Reads the arrays through the dataset manifest."""

    def dense(self, dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        return load_dense_training_arrays(dataset_dir)

    def parameters(self, dataset_dir: Path) -> tuple[np.ndarray, ...]:
        return load_parameter_arrays(dataset_dir)

    def matrix_sample_index(self, dataset_dir: Path) -> np.ndarray:
        return load_matrix_sample_index(dataset_dir)

    def scaled_matrix(self, dataset_dir: Path, row: int, matrix_id: int) -> csr_array:
        artifact = resolve_dataset_artifacts(dataset_dir).matrix
        bindings = load_matrix_sample_index(dataset_dir)
        physical = matrix_row(artifact, bindings, row_index=row, dataset_dir=dataset_dir)
        return load_matrix_sparse_sample(dataset_dir, physical)
