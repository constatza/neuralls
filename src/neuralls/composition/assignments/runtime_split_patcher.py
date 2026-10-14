"""Point a lower-case DLKit job at its dataset's pinned split."""

from __future__ import annotations

from pathlib import Path

from neuralls.composition.assignments._job_types import TrainableJobConfig
from neuralls.shared.constants import SPLIT_FILENAME
from neuralls.shared.errors import StageInputsStale


def patch_split_settings[T: TrainableJobConfig](
    settings: T,
    *,
    dataset_dir: Path,
) -> T:
    """Point settings at the split file generation pinned for ``dataset_dir``.

    Every job trained, searched, fitted, or evaluated against this dataset sees
    the identical held-out partition because they all read the one file. The
    job's own `data.splits` ratios are not consulted: the partition belongs to
    the dataset config's `[split]` section.

    Raises:
        StageInputsStale: If generation has not pinned a split for the dataset.
    """
    pinned_path = dataset_dir / SPLIT_FILENAME
    if not pinned_path.exists():
        raise StageInputsStale(
            f"Dataset '{dataset_dir}' has no pinned split ({SPLIT_FILENAME}): "
            "run `neuralls generate` (or `neuralls run`) first."
        )
    return settings.patch({"data": {"splits": {"filepath": pinned_path}}})
