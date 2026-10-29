"""Filesystem adapter for a dataset's pinned split file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dlkit.infrastructure.io.index import load_split_indices, save_split_indices
from dlkit.infrastructure.types.split import IndexSplit

from neuralls.shared.constants import SPLIT_FILENAME
from neuralls.shared.errors import StageInputsStale


@dataclass(frozen=True, slots=True)
class FsSplitStore:
    """Keeps the split as ``SPLIT_FILENAME`` inside the dataset directory."""

    def read(self, dataset_dir: Path) -> IndexSplit | None:
        split_path = dataset_dir / SPLIT_FILENAME
        return load_split_indices(split_path) if split_path.exists() else None

    def write(self, dataset_dir: Path, split: IndexSplit) -> Path:
        split_path = dataset_dir / SPLIT_FILENAME
        save_split_indices(split, split_path)
        return split_path

    def require(self, dataset_dir: Path) -> Path:
        split_path = dataset_dir / SPLIT_FILENAME
        if not split_path.exists():
            raise StageInputsStale(
                f"Dataset '{dataset_dir}' has no pinned split ({SPLIT_FILENAME}): "
                "run `neuralls generate` (or `neuralls run`) first."
            )
        return split_path
