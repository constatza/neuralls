"""In-memory ``SplitStore``: the same contract as ``FsSplitStore`` without a file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dlkit.infrastructure.types.split import IndexSplit

from neuralls.shared.constants import SPLIT_FILENAME
from neuralls.shared.errors import StageInputsStale


@dataclass(slots=True)
class InMemorySplitStore:
    splits: dict[Path, IndexSplit] = field(default_factory=dict)
    writes: list[Path] = field(default_factory=list)

    def read(self, dataset_dir: Path) -> IndexSplit | None:
        return self.splits.get(dataset_dir)

    def write(self, dataset_dir: Path, split: IndexSplit) -> Path:
        self.splits[dataset_dir] = split
        self.writes.append(dataset_dir)
        return dataset_dir / SPLIT_FILENAME

    def require(self, dataset_dir: Path) -> Path:
        if dataset_dir not in self.splits:
            raise StageInputsStale(f"Dataset '{dataset_dir}' has no pinned split")
        return dataset_dir / SPLIT_FILENAME
