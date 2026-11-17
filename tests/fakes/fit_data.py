"""In-memory ``FitDataLoader`` that records which datasets were read."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array


@dataclass(slots=True)
class FakeFitData:
    """Serves the same arrays for every dataset directory."""

    rhs: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    solutions: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    params: tuple[np.ndarray, ...] = ()
    matrix_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    matrices: dict[int, csr_array] = field(default_factory=dict)
    dense_reads: list[Path] = field(default_factory=list)
    parameter_reads: list[Path] = field(default_factory=list)

    def dense(self, dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        self.dense_reads.append(dataset_dir)
        return self.rhs, self.solutions

    def parameters(self, dataset_dir: Path) -> tuple[np.ndarray, ...]:
        self.parameter_reads.append(dataset_dir)
        return self.params

    def matrix_sample_index(self, dataset_dir: Path) -> np.ndarray:
        del dataset_dir
        return self.matrix_index

    def scaled_matrix(self, dataset_dir: Path, row: int, matrix_id: int) -> csr_array:
        del dataset_dir, row
        return self.matrices[matrix_id]
