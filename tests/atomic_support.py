"""Seeded dense-generation fixtures shared by composition and io/generation conftests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec

_ATOMIC_SEED = 2024
_ATOMIC_MATRIX_COUNT = 3
_ATOMIC_MATRIX_SIZE = 4
_ATOMIC_BATCH_ROWS = 4
"""Small batch size so the streamed run spans several batches."""


@pytest.fixture
def atomic_matrix_dir(tmp_path: Path) -> Path:
    """Seeded SPD matrices as individually globbable .txt files."""
    rng = np.random.default_rng(_ATOMIC_SEED)
    matrix_dir = tmp_path / "atomic_matrices"
    matrix_dir.mkdir()
    for index in range(_ATOMIC_MATRIX_COUNT):
        base = rng.standard_normal((_ATOMIC_MATRIX_SIZE, _ATOMIC_MATRIX_SIZE))
        matrix = base @ base.T + 5.0 * np.eye(_ATOMIC_MATRIX_SIZE)
        np.savetxt(matrix_dir / f"A_{index:03d}.txt", matrix)
    return matrix_dir


@pytest.fixture
def atomic_source(atomic_matrix_dir: Path) -> SourceSpec:
    return SourceSpec(matrix_path=str(atomic_matrix_dir / "A_*.txt"))


@pytest.fixture
def atomic_spec() -> DatasetSpec:
    """Dense streamed spec: forward-only strategy, several batches, fixed seed."""
    return DatasetSpec(
        mixture=MixtureSpec(
            counts={"gaussian_forward": 8},
            seed=_ATOMIC_SEED,
            shuffle=False,
        ),
        normalize="matrix",
        write_batch_size=_ATOMIC_BATCH_ROWS,
    )
