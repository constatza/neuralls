"""Read a dataset's matrix shape and its pinned held-out fold from disk."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from dlkit.infrastructure.io.index import load_split_indices

from neuralls.domain.dataset_shape import DatasetShape, HeldOutFold, MultiMatrix, SingleMatrix
from neuralls.platform.storage.dataset_readers import (
    load_matrix_sample_index,
    resolve_dataset_artifacts,
)
from neuralls.shared.constants import SPLIT_FILENAME
from neuralls.shared.errors import StageInputsStale
from neuralls.shared.types import MatrixId


def classify_dataset(dataset_dir: Path) -> DatasetShape:
    """Classify a dataset as one matrix or several, from its per-row matrix bindings.

    The single place this is decided.
    """
    matrix_ids = tuple(MatrixId(int(i)) for i in np.unique(load_matrix_sample_index(dataset_dir)))
    if len(matrix_ids) == 1:
        return SingleMatrix(matrix_id=matrix_ids[0])
    return MultiMatrix(matrix_ids=matrix_ids)


def load_held_out_fold(dataset_dir: Path) -> HeldOutFold:
    """Group the pinned split's test rows by the matrix each is bound to.

    Raises:
        StageInputsStale: If generation has not pinned a split for the dataset.
    """
    split_path = dataset_dir / SPLIT_FILENAME
    if not split_path.exists():
        raise StageInputsStale(
            f"Dataset '{dataset_dir}' has no pinned split ({SPLIT_FILENAME}): "
            "run `neuralls generate` (or `neuralls run`) first."
        )
    bindings = load_matrix_sample_index(dataset_dir)
    rows_by_matrix: defaultdict[MatrixId, list[int]] = defaultdict(list)
    for row in load_split_indices(split_path).test:
        rows_by_matrix[MatrixId(int(bindings[row]))].append(row)
    return HeldOutFold(rows_by_matrix={m: tuple(rows) for m, rows in rows_by_matrix.items()})


def system_dimension(dataset_dir: Path) -> int:
    """Order of the dataset's linear systems, read from the manifest's RHS shape.

    The RHS length is format-independent, unlike the matrix artifact's shape,
    which differs between dense, broadcast and CSR storage.
    """
    return int(resolve_dataset_artifacts(dataset_dir).rhs.shape[-1])
