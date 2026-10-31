"""`DatasetSplitRecord`: fit and test rows never overlap; matrices may overlap only when shared."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.solver.models.split_record import (
    DatasetSplitRecord,
    MatrixDisjoint,
    SingleMatrixShared,
    build_split_record,
)
from neuralls.shared.types import DatasetId, MatrixId

DATASET = DatasetId("ds")
DIGEST = "sha256:" + "a" * 64


def test_single_matrix_dataset_shares_its_matrix_between_folds() -> None:
    record = build_split_record(
        DATASET,
        DIGEST,
        train=(0, 1, 2),
        validation=(3,),
        test=(4, 5),
        matrix_sample_index=np.zeros(6, dtype=np.int64),
    )

    assert record == DatasetSplitRecord(
        dataset=DATASET,
        split_digest=DIGEST,
        train_rows=3,
        validation_rows=1,
        test_rows=2,
        fit_matrix_ids=(MatrixId(0),),
        test_matrix_ids=(MatrixId(0),),
        separation=SingleMatrixShared(MatrixId(0)),
    )


def test_multi_matrix_dataset_holds_matrices_out() -> None:
    record = build_split_record(
        DATASET,
        DIGEST,
        train=(0, 1, 2, 3),
        validation=(4,),
        test=(5, 6, 7),
        matrix_sample_index=np.array([0, 0, 1, 1, 2, 3, 3, 4]),
    )

    assert record.separation == MatrixDisjoint()
    assert record.fit_matrix_ids == (MatrixId(0), MatrixId(1), MatrixId(2))
    assert record.test_matrix_ids == (MatrixId(3), MatrixId(4))


def test_a_row_in_both_fit_and_test_is_rejected() -> None:
    with pytest.raises(ValueError, match="rows"):
        build_split_record(
            DATASET,
            DIGEST,
            train=(0, 1),
            validation=(),
            test=(1, 2),
            matrix_sample_index=np.zeros(3, dtype=np.int64),
        )


def test_a_validation_row_in_test_is_rejected() -> None:
    with pytest.raises(ValueError, match="rows"):
        build_split_record(
            DATASET,
            DIGEST,
            train=(0,),
            validation=(1,),
            test=(1,),
            matrix_sample_index=np.zeros(2, dtype=np.int64),
        )


def test_a_matrix_in_both_folds_of_a_multi_matrix_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="matrix"):
        build_split_record(
            DATASET,
            DIGEST,
            train=(0, 2),
            validation=(),
            test=(1, 3),
            matrix_sample_index=np.array([0, 0, 1, 1]),
        )


def test_a_row_outside_the_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        build_split_record(
            DATASET,
            DIGEST,
            train=(0,),
            validation=(),
            test=(9,),
            matrix_sample_index=np.zeros(2, dtype=np.int64),
        )
