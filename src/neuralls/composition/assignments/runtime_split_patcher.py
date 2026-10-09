"""Runtime train/validation/test split pinning for lower-case DLKit jobs."""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.config.job_config import FitJobConfig

from neuralls.composition.assignments._job_types import TrainableJobConfig
from neuralls.composition.assignments._split_pinning import resolve_or_create_pinned_split
from neuralls.shared.constants import DEFAULT_SPLIT_SEED

# A one-shot Fittable.fit() call has no training loop to serve a validation
# fold, so the val ratio used for pinning is forced to zero for FitJobConfig
# regardless of what the config carries.
FIT_JOB_VAL_RATIO = 0.0


def patch_split_settings[T: TrainableJobConfig](
    settings: T,
    *,
    dataset_dir: Path,
) -> T:
    """Pin the dataset's train/validation/test split and point settings at it.

    Reuses whichever split file already exists for ``dataset_dir`` (computing
    and persisting one on first use) so every job trained, searched, fitted,
    or evaluated against this dataset sees the identical held-out partition.
    """
    data_cfg = settings.data
    test_ratio = data_cfg.splits.test_ratio
    val_ratio = (
        FIT_JOB_VAL_RATIO if isinstance(settings, FitJobConfig) else data_cfg.splits.val_ratio
    )

    pinned_path = resolve_or_create_pinned_split(
        dataset_dir,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=DEFAULT_SPLIT_SEED,
    )
    return settings.patch({"data": {"splits": {"filepath": pinned_path}}})
