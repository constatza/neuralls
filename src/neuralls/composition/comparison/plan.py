"""Plan every linear system a case will solve, before anything is resolved or run.

A comparison only ever selects systems out of an existing, pinned test fold. The
planner is where that selection happens, once: each ``[[comparisons]]`` /
``[[comparison_sweeps]]`` entry becomes concrete `PlannedSystem` values that
carry their matrix, their RHS row (for the ``dataset`` kind), their derived RHS
seed and their leakage provenance. Everything downstream (config resolution,
the runner, group discovery) consumes the plan and never re-derives which matrix
or row a system means.

Shape rule (decided by `classify_dataset`, the only place that is decided):

* ``[[comparisons]]`` needs a single-matrix dataset;
* ``[[comparison_sweeps]]`` needs a multi-matrix dataset.

Every system solves on a matrix of its own ``matrix_dataset``; an external RHS
file is applied to that native matrix and must match its dimension.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neuralls.composition.comparison.config_assembler import resolve_dataset_dir
from neuralls.composition.comparison.models import MatrixRef, PlannedSystem
from neuralls.domain.dataset_shape import DatasetShape, HeldOutFold, MultiMatrix, SingleMatrix
from neuralls.domain.solver.models.provenance import HeldOutMatrix, SharedSingleMatrix
from neuralls.platform.config.models.comparison import RhsSourceModelBase
from neuralls.platform.config.models.experiments import (
    CaseConfig,
    ComparisonDataBinding,
    ComparisonRegistryEntry,
    ComparisonSweepEntry,
    FirstNSelector,
    FractionSelector,
    TestSetSelector,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.dataset_shape import (
    classify_dataset,
    load_held_out_fold,
    system_dimension,
)
from neuralls.shared.types import ComparisonRhsSourceKind, MatrixId

__all__ = ["derive_seed", "plan_comparisons", "plan_systems"]

_SINGLE_MATRIX_ROW = 0
"""A single-matrix dataset stores its one matrix once (broadcast layout), at row 0."""


def derive_seed(base: int | None, matrix_id: MatrixId, ordinal: int) -> int:
    """Independent, reproducible RHS seed for one (matrix, draw) of a case.

    Hashing the triple through `SeedSequence` makes neighbouring matrices and
    neighbouring draws statistically independent, where reusing one seed (or
    adding the ordinal to it) would hand every system the same or a correlated vector.
    """
    return int(np.random.SeedSequence([base or 0, matrix_id, ordinal]).generate_state(1)[0])


def _system_seed(
    rhs_source: RhsSourceModelBase, base: int | None, matrix_id: MatrixId, ordinal: int
) -> int | None:
    """The RHS seed one system gets: derived per draw if the source consumes one, else the entry's."""
    return derive_seed(base, matrix_id, ordinal) if rhs_source.needs_own_seed else base


@dataclass(slots=True)
class _DatasetCatalog:
    """Reads each dataset's shape, held-out fold and dimension at most once."""

    dirs: Mapping[str, Path]
    _shapes: dict[str, DatasetShape] = field(default_factory=dict)
    _folds: dict[str, HeldOutFold] = field(default_factory=dict)

    def shape(self, dataset_id: str) -> DatasetShape:
        if dataset_id not in self._shapes:
            self._shapes[dataset_id] = classify_dataset(self.dirs[dataset_id])
        return self._shapes[dataset_id]

    def fold(self, dataset_id: str) -> HeldOutFold:
        if dataset_id not in self._folds:
            self._folds[dataset_id] = load_held_out_fold(self.dirs[dataset_id])
        return self._folds[dataset_id]

    def require_uniform_dimension(self, dataset_ids: Sequence[str]) -> None:
        """Reject datasets whose systems differ in size: only same-size systems are compared."""
        dimensions = {
            dataset_id: system_dimension(self.dirs[dataset_id])
            for dataset_id in dict.fromkeys(dataset_ids)
        }
        if len(set(dimensions.values())) > 1:
            raise ValueError(
                "A case may only compare systems of one size, but its datasets have "
                f"different matrix dimensions: {dimensions}."
            )


def _concrete_entry(
    source: ComparisonDataBinding,
    *,
    entry_id: str,
    seed: int | None,
    display_name: str | None,
) -> ComparisonRegistryEntry:
    return ComparisonRegistryEntry(
        id=entry_id,
        matrix_dataset=source.matrix_dataset,
        method=source.method,
        rhs_source=source.rhs_source,
        rhs_count=source.rhs_count,
        require_non_residual_rhs=source.require_non_residual_rhs,
        seed=seed,
        assignments=source.assignments,
        display_name=display_name,
    )


def _ordinal_suffix(rhs_count: int, ordinal: int) -> str:
    return f"-{ordinal}" if rhs_count > 1 else ""


def _require_rhs_count_fits(source: ComparisonDataBinding, owner: str) -> None:
    if source.rhs_source.single_draw and source.rhs_count != 1:
        raise ValueError(
            f"{owner}: rhs_source '{source.rhs_source.kind}' is one fixed vector, "
            f"so rhs_count must be 1, got {source.rhs_count}."
        )


@dataclass(frozen=True, slots=True)
class _MatrixSlot:
    """A matrix to plan systems on, with the held-out rows bound to it."""

    matrix_id: MatrixId
    first_row: int
    held_out_rows: tuple[int, ...]


def _systems_on_matrix(
    source: ComparisonDataBinding,
    *,
    owner: str,
    id_stem: str,
    display_stem: str | None,
    slot: _MatrixSlot,
    provenance: SharedSingleMatrix | HeldOutMatrix,
    matrix_row_follows_rhs_row: bool,
) -> list[PlannedSystem]:
    """Plan the `rhs_count` systems of one source on one matrix."""
    kind = ComparisonRhsSourceKind(source.rhs_source.kind)
    if kind is ComparisonRhsSourceKind.DATASET and len(slot.held_out_rows) < source.rhs_count:
        raise ValueError(
            f"{owner}: rhs_count={source.rhs_count} but matrix {slot.matrix_id} of dataset "
            f"'{source.matrix_dataset}' has only {len(slot.held_out_rows)} held-out rows."
        )
    systems: list[PlannedSystem] = []
    for ordinal in range(source.rhs_count):
        rhs_row = slot.held_out_rows[ordinal] if kind is ComparisonRhsSourceKind.DATASET else None
        seed = _system_seed(source.rhs_source, source.seed, slot.matrix_id, ordinal)
        row = rhs_row if matrix_row_follows_rhs_row and rhs_row is not None else slot.first_row
        systems.append(
            PlannedSystem(
                entry=_concrete_entry(
                    source,
                    entry_id=f"{id_stem}{_ordinal_suffix(source.rhs_count, ordinal)}",
                    seed=seed,
                    display_name=display_stem,
                ),
                matrix=MatrixRef(matrix_id=slot.matrix_id, row=row),
                rhs_row=rhs_row,
                rhs_ordinal=ordinal,
                provenance=provenance,
            )
        )
    return systems


def _plan_comparison(
    entry: ComparisonRegistryEntry, catalog: _DatasetCatalog
) -> list[PlannedSystem]:
    owner = f"[[comparisons]] entry '{entry.id}'"
    _require_rhs_count_fits(entry, owner)
    shape = catalog.shape(entry.matrix_dataset)
    match shape:
        case MultiMatrix(matrix_ids=matrix_ids):
            raise ValueError(
                f"{owner}: dataset '{entry.matrix_dataset}' holds {len(matrix_ids)} matrices, "
                "but [[comparisons]] needs a single-matrix dataset. Use [[comparison_sweeps]] "
                "for multi-matrix datasets."
            )
        case SingleMatrix(matrix_id=matrix_id):
            held_out_rows = (
                catalog.fold(entry.matrix_dataset).rows_by_matrix.get(matrix_id, ())
                if entry.rhs_source.kind == ComparisonRhsSourceKind.DATASET
                else ()
            )
            return _systems_on_matrix(
                entry,
                owner=owner,
                id_stem=entry.id,
                display_stem=entry.display_name,
                slot=_MatrixSlot(matrix_id, _SINGLE_MATRIX_ROW, held_out_rows),
                provenance=SharedSingleMatrix(raw_matrix_index=_SINGLE_MATRIX_ROW),
                matrix_row_follows_rhs_row=False,
            )


def _select_matrices(
    selector: TestSetSelector, held_out: tuple[MatrixId, ...], owner: str
) -> tuple[MatrixId, ...]:
    """Choose among the held-out matrices; an empty choice is an error, never a silent no-op."""
    match selector:
        case FirstNSelector(n=n):
            chosen = held_out[:n]
        case FractionSelector(p=p, seed=seed):
            count = round(p * len(held_out))
            positions = np.random.default_rng(seed).choice(len(held_out), size=count, replace=False)
            chosen = tuple(held_out[position] for position in sorted(int(c) for c in positions))
    if not chosen:
        raise ValueError(
            f"{owner}: selector {selector.model_dump()} selects no matrices out of the "
            f"{len(held_out)} held-out ones."
        )
    return chosen


def _plan_sweep(sweep: ComparisonSweepEntry, catalog: _DatasetCatalog) -> list[PlannedSystem]:
    owner = f"[[comparison_sweeps]] entry '{sweep.id_prefix}'"
    _require_rhs_count_fits(sweep, owner)
    shape = catalog.shape(sweep.matrix_dataset)
    match shape:
        case SingleMatrix():
            raise ValueError(
                f"{owner}: dataset '{sweep.matrix_dataset}' holds a single matrix, but "
                "[[comparison_sweeps]] needs a multi-matrix dataset. Use [[comparisons]] "
                "for single-matrix datasets."
            )
        case MultiMatrix():
            pass
    fold = catalog.fold(sweep.matrix_dataset)
    systems: list[PlannedSystem] = []
    for matrix_id in _select_matrices(sweep.selector, fold.matrices, owner):
        rows = fold.rows_by_matrix[matrix_id]
        systems.extend(
            _systems_on_matrix(
                sweep,
                owner=owner,
                id_stem=f"{sweep.id_prefix}-{matrix_id}",
                display_stem=sweep.display_name,
                slot=_MatrixSlot(matrix_id, rows[0], rows),
                provenance=HeldOutMatrix(matrix_id=matrix_id, raw_matrix_index=rows[0]),
                matrix_row_follows_rhs_row=True,
            )
        )
    return systems


def _reject_duplicate_ids(systems: Sequence[PlannedSystem]) -> None:
    counts = Counter(system.entry.id for system in systems)
    duplicates = sorted(system_id for system_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Planned comparison ids are not unique: {', '.join(duplicates)}. "
            "Give the entries distinct ids / id_prefixes."
        )


def plan_systems(
    case_cfg: CaseConfig, dataset_dirs: Mapping[str, Path]
) -> tuple[PlannedSystem, ...]:
    """Plan every system of a case against already-resolved dataset directories.

    Args:
        case_cfg: The validated case config.
        dataset_dirs: Dataset registry id -> generated dataset directory, covering
            every ``matrix_dataset`` the case's entries reference.

    Returns:
        Planned systems: every ``[[comparisons]]`` entry's, then every sweep's, in
        declared order.

    Raises:
        ValueError: On a shape-rule violation, an unsatisfiable ``rhs_count`` or
            selector, duplicate planned ids, or datasets of different system size.
        StageInputsStale: If a needed dataset has no pinned split.
    """
    catalog = _DatasetCatalog(dirs=dataset_dirs)
    systems = [
        *(s for entry in case_cfg.comparisons for s in _plan_comparison(entry, catalog)),
        *(s for sweep in case_cfg.comparison_sweeps for s in _plan_sweep(sweep, catalog)),
    ]
    _reject_duplicate_ids(systems)
    catalog.require_uniform_dimension([system.entry.matrix_dataset for system in systems])
    return tuple(systems)


def plan_comparisons(
    case_cfg: CaseConfig, config_dir: Path, settings: NeurallsSettings
) -> tuple[PlannedSystem, ...]:
    """Plan a case's systems; run after the input gate has proven its datasets current."""
    dataset_ids = dict.fromkeys(
        entry.matrix_dataset for entry in (*case_cfg.comparisons, *case_cfg.comparison_sweeps)
    )
    return plan_systems(
        case_cfg,
        {
            dataset_id: resolve_dataset_dir(case_cfg, config_dir, dataset_id, settings)
            for dataset_id in dataset_ids
        },
    )
