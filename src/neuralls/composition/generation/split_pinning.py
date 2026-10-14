"""Compute and persist a dataset's train/validation/test split during generation.

dlkit's ``RatioSplitStrategy`` produces a fresh random split every time it
runs. Left unpinned, two training runs against the "same" dataset directory
could see different held-out rows. Generation therefore owns the partition: it
derives it from the dataset's matrix bindings and the `[split]` config, and
persists it next to the dataset. Training, evaluation and comparison only read
the file (via dlkit's ``ExternalFileSplitStrategy`` or ``load_split_indices``);
none of them creates or alters it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dlkit.infrastructure.io.index import load_split_indices, save_split_indices
from dlkit.infrastructure.types.split import IndexSplit, RatioSplitStrategy
from loguru import logger

from neuralls.platform.config.models.data_models import SplitConfig
from neuralls.platform.storage.dataset_readers import load_matrix_sample_index
from neuralls.shared.constants import SPLIT_FILENAME


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


def expected_split(dataset_dir: Path, cfg: SplitConfig) -> IndexSplit:
    """The split `cfg` yields for this dataset: deterministic in bindings, ratios and seed."""
    return _compute_split(
        load_matrix_sample_index(dataset_dir),
        test_ratio=cfg.test_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )


def split_is_current(dataset_dir: Path, cfg: SplitConfig) -> bool:
    """Whether the dataset's split file exists and equals what `cfg` yields now. Read-only."""
    split_path = dataset_dir / SPLIT_FILENAME
    return split_path.exists() and load_split_indices(split_path) == expected_split(
        dataset_dir, cfg
    )


def pin_dataset_split(dataset_dir: Path, cfg: SplitConfig) -> Path:
    """Write the dataset's split file, replacing it only when its content changes.

    The split is a deterministic function of the dataset's matrix bindings and
    `cfg`, so recomputing is cheap and an unchanged result leaves the file (and
    every downstream identity keyed on it) untouched.

    Args:
        dataset_dir: Generated dataset directory.
        cfg: The dataset config's `[split]` section.

    Returns:
        Path to the split file.
    """
    split_path = dataset_dir / SPLIT_FILENAME
    split = expected_split(dataset_dir, cfg)
    existed = split_path.exists()
    if existed and load_split_indices(split_path) == split:
        return split_path
    save_split_indices(split, split_path)
    logger.info(
        "{} split for '{}' (test_ratio={}, val_ratio={}, seed={}): {} train / {} val / {} test rows",
        "Replaced" if existed else "Pinned",
        dataset_dir,
        cfg.test_ratio,
        cfg.val_ratio,
        cfg.seed,
        len(split.train),
        len(split.validation),
        len(split.test),
    )
    return split_path
