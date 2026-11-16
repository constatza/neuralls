"""Which stored rows define the energy-POD reference matrix: train-fold matrices only."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.composition.preconditioners.reference_matrix import train_matrix_rows


@pytest.fixture
def matrix_sample_index() -> np.ndarray:
    """Six rows over three matrices (ids 4, 7, 9), two rows each, interleaved."""
    return np.array([7, 4, 7, 9, 4, 9])


def test_each_train_matrix_is_named_once_by_its_first_train_row(
    matrix_sample_index: np.ndarray,
) -> None:
    assert train_matrix_rows(matrix_sample_index, (4, 2, 0, 1)) == {4: 1, 7: 0}


def test_a_matrix_bound_only_to_held_out_rows_is_excluded(
    matrix_sample_index: np.ndarray,
) -> None:
    assert train_matrix_rows(matrix_sample_index, (0, 2)) == {7: 0}


def test_an_empty_train_fold_names_no_matrix(matrix_sample_index: np.ndarray) -> None:
    assert train_matrix_rows(matrix_sample_index, ()) == {}
