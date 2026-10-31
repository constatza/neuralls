"""What a dataset's pinned split held out, checkable without the dataset directory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from neuralls.shared.digest import Digest
from neuralls.shared.types import DatasetId, MatrixId


@dataclass(frozen=True, slots=True)
class SingleMatrixShared:
    """The dataset's one matrix is in every fold by construction; only RHS rows are held out."""

    matrix_id: MatrixId


@dataclass(frozen=True, slots=True)
class MatrixDisjoint:
    """No matrix of the fit folds appears in the test fold."""


type MatrixSeparation = SingleMatrixShared | MatrixDisjoint


@dataclass(frozen=True, slots=True)
class DatasetSplitRecord:
    """One dataset's pinned split: its digest, fold sizes and the matrices on each side.

    Attributes:
        dataset: Dataset registry id.
        split_digest: Content digest of the pinned split file.
        train_rows: Rows in the train fold.
        validation_rows: Rows in the validation fold.
        test_rows: Rows in the test fold.
        fit_matrix_ids: Matrices with a row in the train or validation fold, ascending.
        test_matrix_ids: Matrices with a row in the test fold, ascending.
        separation: How the two matrix sets relate; checked against them on construction.
    """

    dataset: DatasetId
    split_digest: Digest
    train_rows: int
    validation_rows: int
    test_rows: int
    fit_matrix_ids: tuple[MatrixId, ...]
    test_matrix_ids: tuple[MatrixId, ...]
    separation: MatrixSeparation

    def __post_init__(self) -> None:
        """Reject a separation claim the matrix sets contradict."""
        match self.separation:
            case SingleMatrixShared(matrix_id=shared):
                if not set(self.fit_matrix_ids) | set(self.test_matrix_ids) <= {shared}:
                    raise ValueError(f"single-matrix dataset {self.dataset!r} lists other matrices")
            case MatrixDisjoint():
                overlap = set(self.fit_matrix_ids) & set(self.test_matrix_ids)
                if overlap:
                    raise ValueError(
                        f"matrix ids {sorted(overlap)} of dataset {self.dataset!r} are in both "
                        "the fit and test folds"
                    )


def build_split_record(
    dataset: DatasetId,
    split_digest: Digest,
    *,
    train: Sequence[int],
    validation: Sequence[int],
    test: Sequence[int],
    matrix_sample_index: np.ndarray,
) -> DatasetSplitRecord:
    """Describe a pinned split by the matrices each side holds.

    Args:
        dataset: Dataset registry id.
        split_digest: Content digest of the pinned split file.
        train: Row indices of the train fold.
        validation: Row indices of the validation fold.
        test: Row indices of the test fold.
        matrix_sample_index: Matrix id of every dataset row.

    Returns:
        The split record.

    Raises:
        ValueError: A row index lies outside the dataset, a row is in both a fit fold and the
            test fold, or a matrix of a multi-matrix dataset is in both.
    """
    n_rows = matrix_sample_index.size
    for row in (*train, *validation, *test):
        if not 0 <= row < n_rows:
            raise ValueError(f"row {row} of dataset {dataset!r} is outside its {n_rows} rows")
    fit_rows = {*train, *validation}
    shared_rows = fit_rows & set(test)
    if shared_rows:
        raise ValueError(
            f"rows {sorted(shared_rows)[:5]} of dataset {dataset!r} are in both the fit and test folds"
        )
    matrix_ids = np.unique(matrix_sample_index)
    separation: MatrixSeparation = (
        SingleMatrixShared(MatrixId(int(matrix_ids[0])))
        if matrix_ids.size == 1
        else MatrixDisjoint()
    )
    return DatasetSplitRecord(
        dataset=dataset,
        split_digest=split_digest,
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        fit_matrix_ids=_matrices_of(fit_rows, matrix_sample_index),
        test_matrix_ids=_matrices_of(set(test), matrix_sample_index),
        separation=separation,
    )


def _matrices_of(rows: set[int], matrix_sample_index: np.ndarray) -> tuple[MatrixId, ...]:
    ids = np.unique(matrix_sample_index[sorted(rows)]) if rows else np.empty(0, dtype=np.int64)
    return tuple(MatrixId(int(matrix_id)) for matrix_id in ids)
