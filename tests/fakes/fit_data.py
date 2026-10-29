"""In-memory ``FitDataLoader`` that records which datasets were read."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class FakeFitData:
    """Serves the same arrays for every dataset directory."""

    rhs: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    solutions: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    params: tuple[np.ndarray, ...] = ()
    dense_reads: list[Path] = field(default_factory=list)
    parameter_reads: list[Path] = field(default_factory=list)

    def dense(self, dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        self.dense_reads.append(dataset_dir)
        return self.rhs, self.solutions

    def parameters(self, dataset_dir: Path) -> tuple[np.ndarray, ...]:
        self.parameter_reads.append(dataset_dir)
        return self.params
