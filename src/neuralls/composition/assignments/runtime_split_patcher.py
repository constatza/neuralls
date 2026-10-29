"""Point a lower-case DLKit job at its dataset's pinned split."""

from __future__ import annotations

from pathlib import Path

from neuralls.composition.assignments._job_types import TrainableJobConfig


def patch_split_settings[T: TrainableJobConfig](settings: T, *, split_file: Path) -> T:
    """Point settings at the split file generation pinned.

    Every job trained, searched, fitted, or evaluated against a dataset sees
    the identical held-out partition because they all read the one file. The
    job's own `data.splits` ratios are not consulted: the partition belongs to
    the dataset config's `[split]` section.
    """
    return settings.patch({"data": {"splits": {"filepath": split_file}}})
