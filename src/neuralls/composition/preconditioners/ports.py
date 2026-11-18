"""Contracts the preconditioner builders consume; adapters live in ``platform``."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.sparse import csr_array


class FitDataLoader(Protocol):
    """Reads the arrays a fitted coarsening trains on from a generated dataset."""

    def dense(self, dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """The dataset's right-hand sides and solutions."""
        ...

    def parameters(self, dataset_dir: Path) -> tuple[np.ndarray, ...]:
        """The dataset's parameter arrays, in manifest order."""
        ...

    def matrix_sample_index(self, dataset_dir: Path) -> np.ndarray:
        """The matrix id every dataset row is bound to."""
        ...

    def scaled_matrix(self, dataset_dir: Path, row: int, matrix_id: int) -> csr_array:
        """The stored (scaled) matrix ``matrix_id`` that dataset row ``row`` binds."""
        ...
