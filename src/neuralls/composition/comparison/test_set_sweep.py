"""Expand one ``[[comparison_sweeps]]`` entry into concrete comparison entries.

A sweep entry names no single test-fold position itself; `expand_test_set_sweep`
resolves its `TestSetSelector` against the sweep's own dataset's pinned
`split.json` (via `pinned_split.py`, Phase B's file) and returns one ordinary
`ComparisonRegistryEntry` per resolved position, each carrying the entry's
`test_position` set — indistinguishable downstream from a hand-written
`[[comparisons]]` entry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuralls.composition.comparison.pinned_split import (
    resolve_test_matrix_indices,
    resolve_test_row_indices,
)
from neuralls.platform.config.models.experiments import (
    AllSelector,
    ComparisonRegistryEntry,
    ComparisonSweepEntry,
    FirstNSelector,
    FractionSelector,
    SubsetSelector,
    TestSetSelector,
)
from neuralls.shared.types import ComparisonRhsSourceKind, TestPosition


def _fold_length(sweep_entry: ComparisonSweepEntry, *, dataset_dir: Path) -> int:
    """Return how many test-fold positions are addressable for this sweep's RHS kind.

    Mirrors the row-vs-matrix dispatch `comparison_batch.py` already performs
    for a single `test_position`: the `dataset` RHS kind addresses pinned-split
    rows directly, every other kind addresses the matrices those rows bind to.
    """
    if sweep_entry.rhs_source.kind == ComparisonRhsSourceKind.DATASET:
        return len(resolve_test_row_indices(dataset_dir))
    return len(resolve_test_matrix_indices(dataset_dir))


def _resolve_positions(selector: TestSetSelector, *, fold_length: int) -> list[TestPosition]:
    """Resolve a selector into concrete, position-ordered `TestPosition` values."""
    match selector:
        case FractionSelector(p=p, seed=seed):
            rng = np.random.default_rng(seed)
            count = round(p * fold_length)
            chosen = rng.choice(fold_length, size=count, replace=False)
            return [TestPosition(position) for position in sorted(int(c) for c in chosen)]
        case AllSelector():
            return [TestPosition(position) for position in range(fold_length)]
        case FirstNSelector(n=n):
            return [TestPosition(position) for position in range(min(n, fold_length))]
        case SubsetSelector(positions=positions):
            return [TestPosition(position) for position in positions]


def expand_test_set_sweep(
    sweep_entry: ComparisonSweepEntry, *, dataset_dir: Path
) -> list[ComparisonRegistryEntry]:
    """Expand one sweep entry into one `ComparisonRegistryEntry` per resolved position.

    Args:
        sweep_entry: The `[[comparison_sweeps]]` entry to expand.
        dataset_dir: Directory of `sweep_entry.matrix_dataset`, already resolved
            by the caller (`composition/comparison/config_assembler.py::resolve_dataset_dir`).

    Returns:
        Expanded entries in resolved-position order, each with a stable
        ``f"{id_prefix}-{position}"`` id and `test_position` set.
    """
    fold_length = _fold_length(sweep_entry, dataset_dir=dataset_dir)
    positions = _resolve_positions(sweep_entry.selector, fold_length=fold_length)
    return [
        ComparisonRegistryEntry(
            id=f"{sweep_entry.id_prefix}-{position}",
            matrix_dataset=sweep_entry.matrix_dataset,
            test_position=position,
            method=sweep_entry.method,
            rhs_source=sweep_entry.rhs_source,
            require_non_residual_rhs=sweep_entry.require_non_residual_rhs,
            seed=sweep_entry.seed,
            assignments=sweep_entry.assignments,
            display_name=sweep_entry.display_name,
        )
        for position in positions
    ]
