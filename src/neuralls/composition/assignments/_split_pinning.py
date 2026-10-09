"""Compute and persist a dataset's train/validation/test split exactly once.

dlkit's ``RatioSplitStrategy`` produces a fresh random row-level split every
time it runs. Left unpinned, that means two training runs against the "same"
dataset directory can silently see different held-out rows. This module
computes the split once per dataset directory and persists it next to the
dataset, so every subsequent job that trains, searches, fits, or evaluates
against that dataset reuses the identical partition via dlkit's
``ExternalFileSplitStrategy``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dlkit.infrastructure.io.index import save_split_indices
from dlkit.infrastructure.types.split import IndexSplit, RatioSplitStrategy

from neuralls.platform.storage.dataset_readers import load_matrix_sample_index

SPLIT_FILENAME = "split.json"


def _expand_group_split_to_rows(
    group_split: IndexSplit,
    *,
    row_indices_by_group: list[np.ndarray],
) -> IndexSplit:
    def _rows_for_groups(group_ids: tuple[int, ...]) -> tuple[int, ...]:
        if not group_ids:
            return ()
        rows = np.concatenate([row_indices_by_group[group_id] for group_id in group_ids])
        return tuple(int(row) for row in np.sort(rows))

    predict_groups = group_split.predict
    return IndexSplit(
        train=_rows_for_groups(group_split.train),
        validation=_rows_for_groups(group_split.validation),
        test=_rows_for_groups(group_split.test),
        predict=_rows_for_groups(predict_groups) if predict_groups is not None else None,
    )


def _compute_split(
    matrix_sample_index: np.ndarray,
    *,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> IndexSplit:
    unique_matrix_ids = np.unique(matrix_sample_index)
    if unique_matrix_ids.size == 1:
        return RatioSplitStrategy(
            num_samples=len(matrix_sample_index),
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed,
        ).split()

    # Many-matrix dataset: a plain row-level split could place some rows of
    # one matrix (its multiple RHS samples) in train and others in test,
    # which does not actually hold that matrix out. Split at the matrix-group
    # level instead, then expand each group's chosen fold back to every row
    # sharing that matrix id, so a matrix's rows always land together.
    row_indices_by_group = [
        np.flatnonzero(matrix_sample_index == matrix_id) for matrix_id in unique_matrix_ids
    ]
    group_split = RatioSplitStrategy(
        num_samples=len(unique_matrix_ids),
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
    ).split()
    return _expand_group_split_to_rows(group_split, row_indices_by_group=row_indices_by_group)


def resolve_or_create_pinned_split(
    dataset_dir: Path,
    *,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Path:
    """Return the dataset's pinned split file path, creating it if absent.

    An existing ``split.json`` is trusted and returned unchanged rather than
    recomputed or re-validated, per the "validate once" principle: the split
    was already a valid partition when it was first written, and recomputing
    it with possibly-different ratios would retroactively change what earlier
    runs trained on.
    """
    split_path = dataset_dir / SPLIT_FILENAME
    if split_path.exists():
        return split_path

    matrix_sample_index = load_matrix_sample_index(dataset_dir)
    split = _compute_split(
        matrix_sample_index,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    save_split_indices(split, split_path)
    return split_path
