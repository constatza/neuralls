"""Storage and sampling invariants for multi-matrix generated datasets.

Matrix storage is row-expanded: ``matrix.npy`` holds one normalized matrix per
emitted row, so row ``r`` of ``rhs``/``solutions`` pairs with row ``r`` of the
matrix artifact. ``matrix_sample_index`` names the source matrix id of each row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.dataset_readers import load_matrix_sample_index
from tests.generation.test_orchestration_characterization import spd_matrix_dir  # noqa: F401

RESIDUAL_TOLERANCE = 1e-8
MIXTURE_SEED = 1234


@pytest.fixture
def mixture_dataset(spd_matrix_dir: Path, tmp_path: Path) -> Path:  # noqa: F811
    """Three distinct SPD matrices, two strategies, matrix normalization, npy storage."""
    out_dir = tmp_path / "dataset"
    build_dataset(
        SourceSpec(matrix_path=str(spd_matrix_dir / "A_*.txt")),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": 6, "gaussian_residuals": 6},
                seed=MIXTURE_SEED,
                shuffle=False,
                strategy_overrides={"gaussian_residuals": {"stop": 2, "start": 1}},
            ),
            normalize="matrix",
        ),
        str(out_dir),
        dataset_format="npy",
    )
    return out_dir


def test_each_row_solves_against_its_stored_matrix(mixture_dataset: Path) -> None:
    """Row r satisfies ``S[r] @ solutions[r] == rhs[r]`` within RESIDUAL_TOLERANCE.

    ``S`` is the normalized matrix persisted in ``matrix.npy``. Normalization
    scales each matrix by a positive constant before any RHS is generated from
    it, so the RHS was computed with exactly the stored (normalized) matrix and
    no rescaling is needed to compare.
    """
    stored = np.load(mixture_dataset / "matrix.npy")
    rhs = np.load(mixture_dataset / "rhs.npy")
    solutions = np.load(mixture_dataset / "solutions.npy")
    row_matrix_ids = load_matrix_sample_index(mixture_dataset)

    assert stored.shape[0] == rhs.shape[0] == row_matrix_ids.shape[0]
    assert len(np.unique(row_matrix_ids)) == 3
    for row in range(rhs.shape[0]):
        residual = np.abs(stored[row] @ solutions[row] - rhs[row]).max()
        assert residual <= RESIDUAL_TOLERANCE, f"row {row} residual {residual:.3e}"


def test_rhs_rows_are_distinct_across_strategies(mixture_dataset: Path) -> None:
    """Two strategies sharing a binding must not emit identical RHS vectors.

    Each strategy in a mixture draws from its own stream. A shared seed makes
    gaussian_forward and gaussian_residuals reproduce the same draw, so the
    dataset would contain duplicate training rows.
    """
    rhs = np.load(mixture_dataset / "rhs.npy")
    assert np.unique(rhs, axis=0).shape[0] == rhs.shape[0]
