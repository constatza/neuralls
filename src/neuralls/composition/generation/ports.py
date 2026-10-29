"""Contracts that generation and job materialization consume; adapters live in ``platform``."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dlkit.infrastructure.types.split import IndexSplit


class SplitStore(Protocol):
    """Where a dataset's pinned train/validation/test split lives."""

    def read(self, dataset_dir: Path) -> IndexSplit | None:
        """The pinned split of ``dataset_dir``, or ``None`` when none was pinned."""
        ...

    def write(self, dataset_dir: Path, split: IndexSplit) -> Path:
        """Pin ``split`` for ``dataset_dir``, replacing any earlier one; return the split file."""
        ...

    def require(self, dataset_dir: Path) -> Path:
        """The pinned split file of ``dataset_dir``.

        Raises:
            StageInputsStale: When generation has not pinned a split for the dataset.
        """
        ...
