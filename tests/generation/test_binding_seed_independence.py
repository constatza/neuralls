"""Random streams must be independent across bindings and across strategies.

Each binding is one matrix. A strategy seeded only from the mixture seed would draw the same
raw numbers for every binding with the same sample count. The matrices here are identical on
purpose: any repeat in the output then comes from the random stream, not from the matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.datasets import load_dense_training_arrays

_MATRIX_SIZE = 5
_MIXTURE_SEED = 1234


@pytest.fixture
def identical_matrix_dir(tmp_path: Path) -> Path:
    """Two byte-identical SPD matrices, one ``.txt`` file each."""
    rng = np.random.default_rng(11)
    base = rng.standard_normal((_MATRIX_SIZE, _MATRIX_SIZE))
    matrix = base @ base.T + 5.0 * np.eye(_MATRIX_SIZE)
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    for index in range(2):
        np.savetxt(matrix_dir / f"A_{index:03d}.txt", matrix)
    return matrix_dir


def _build(matrix_dir: Path, out_dir: Path, counts: dict[str, int]) -> Path:
    build_dataset(
        SourceSpec(matrix_path=str(matrix_dir / "A_*.txt")),
        DatasetSpec(mixture=MixtureSpec(counts=counts, seed=_MIXTURE_SEED, shuffle=False)),
        str(out_dir),
        dataset_format="hdf5",
    )
    return out_dir


def test_same_strategy_on_identical_matrices_draws_different_solutions(
    identical_matrix_dir: Path, tmp_path: Path
) -> None:
    """One sample per binding of one strategy: the two bindings must not draw the same vector."""
    dataset = _build(identical_matrix_dir, tmp_path / "dataset", {"gaussian_forward": 2})

    _, solutions = load_dense_training_arrays(dataset)

    assert solutions.shape == (2, _MATRIX_SIZE)
    assert not np.allclose(solutions[0], solutions[1])


def test_two_strategies_on_identical_matrix_draw_different_solutions(
    identical_matrix_dir: Path, tmp_path: Path
) -> None:
    """Two strategies in one mixture must not share a random stream."""
    dataset = _build(
        identical_matrix_dir,
        tmp_path / "dataset",
        {"gaussian_forward": 1, "uniform_forward": 1},
    )

    _, solutions = load_dense_training_arrays(dataset)

    assert solutions.shape == (2, _MATRIX_SIZE)
    assert not np.allclose(solutions[0], solutions[1])
