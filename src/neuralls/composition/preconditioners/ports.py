"""Contracts the preconditioner builders consume; adapters live in ``platform``."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class FitDataLoader(Protocol):
    """Reads the arrays a fitted coarsening trains on from a generated dataset."""

    def dense(self, dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """The dataset's right-hand sides and solutions."""
        ...

    def parameters(self, dataset_dir: Path) -> tuple[np.ndarray, ...]:
        """The dataset's parameter arrays, in manifest order."""
        ...
