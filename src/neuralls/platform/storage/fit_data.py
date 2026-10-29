"""Filesystem adapter that reads fit arrays from a manifest-driven dataset directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuralls.platform.storage.dataset_readers import (
    load_dense_training_arrays,
    load_parameter_arrays,
)


@dataclass(frozen=True, slots=True)
class FsFitDataLoader:
    """Reads the arrays through the dataset manifest."""

    def dense(self, dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        return load_dense_training_arrays(dataset_dir)

    def parameters(self, dataset_dir: Path) -> tuple[np.ndarray, ...]:
        return load_parameter_arrays(dataset_dir)
