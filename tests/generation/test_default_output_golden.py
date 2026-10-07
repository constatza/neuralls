"""Golden digests for the default generation output on a multi-matrix mixture.

Rows are written in generation order: each binding's strategies in mixture order, with no
permutation. The digests of rhs, solutions and row_kind_codes pin that order. The
matrix_sample_index digest pins the per-binding row counts and binding order.
The shuffle=False digests are pinned elsewhere.

Any further digest change is a behavior change to explain, not a value to refresh
casually.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.dataset_readers import (
    load_matrix_sample_index,
    load_row_kind_codes,
)
from neuralls.platform.storage.datasets import load_dense_training_arrays

_FLOAT_DIGEST_DECIMALS = 12
"""Rounding applied before hashing floating arrays, so BLAS/LAPACK last-bit noise
does not flip the digest. Noise is ~1e-15; 1e-12 still catches real regressions."""

_DIGEST_HEX_CHARS = 16
"""Length of the truncated SHA-256 hex digest used as the golden value."""

_SEED = 1234
_MATRIX_COUNT = 3
_MATRIX_SIZE = 5

_EXPECTED_DIGESTS: dict[str, str] = {
    "rhs": "1ed7b231314cb7ca",
    "solutions": "2cf722daad931872",
    "row_kind_codes": "62a7c5d9837d645b",
    "matrix_sample_index": "c3af06584c6a3d1a",
}


def _digest(array: np.ndarray) -> str:
    """Stable content digest of an array's dtype, shape and (rounded) bytes."""
    contiguous = np.ascontiguousarray(array)
    if np.issubdtype(contiguous.dtype, np.floating):
        contiguous = np.round(contiguous, decimals=_FLOAT_DIGEST_DECIMALS)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode())
    hasher.update(str(contiguous.shape).encode())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()[:_DIGEST_HEX_CHARS]


@pytest.fixture
def spd_matrix_dir(tmp_path: Path) -> Path:
    """Three deterministic 5x5 SPD matrices as individually globbable .txt files."""
    rng = np.random.default_rng(11)
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    for index in range(_MATRIX_COUNT):
        base = rng.standard_normal((_MATRIX_SIZE, _MATRIX_SIZE))
        matrix = base @ base.T + 5.0 * np.eye(_MATRIX_SIZE)
        np.savetxt(matrix_dir / f"A_{index:03d}.txt", matrix)
    return matrix_dir


@pytest.fixture
def default_mixture_dataset(
    spd_matrix_dir: Path, tmp_path: Path, solver_overrides: dict[str, Any]
) -> Path:
    """Multi-matrix mixture built through the public entry point with default settings."""
    out_dir = tmp_path / "dataset"
    build_dataset(
        SourceSpec(matrix_path=str(spd_matrix_dir / "A_*.txt")),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"gaussian_forward": 6, "gaussian_residuals": 6},
                seed=_SEED,
                strategy_overrides={"gaussian_residuals": {"stop": 3, "start": 1}},
                solver_overrides=solver_overrides,
            ),
            normalize="matrix",
        ),
        str(out_dir),
        dataset_format="hdf5",
    )
    return out_dir


def test_default_mixture_rhs_digest_is_stable(default_mixture_dataset: Path) -> None:
    rhs, _ = load_dense_training_arrays(default_mixture_dataset)
    assert _digest(rhs) == _EXPECTED_DIGESTS["rhs"]


def test_default_mixture_solutions_digest_is_stable(default_mixture_dataset: Path) -> None:
    _, solutions = load_dense_training_arrays(default_mixture_dataset)
    assert _digest(solutions) == _EXPECTED_DIGESTS["solutions"]


def test_default_mixture_row_kind_codes_digest_is_stable(default_mixture_dataset: Path) -> None:
    assert (
        _digest(load_row_kind_codes(default_mixture_dataset)) == _EXPECTED_DIGESTS["row_kind_codes"]
    )


def test_default_mixture_matrix_sample_index_digest_is_stable(
    default_mixture_dataset: Path,
) -> None:
    """Pins the matrix-index encoding exactly as stored today, including its quirks."""
    assert (
        _digest(load_matrix_sample_index(default_mixture_dataset))
        == _EXPECTED_DIGESTS["matrix_sample_index"]
    )
