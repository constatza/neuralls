"""Data-driven comparison-group discovery and its up-front validations.

Groups are discovered from each entry's own data (dataset, resolved matrix
index, RHS kind), never from which TOML array declared the entry. Pure reads
of dataset metadata — nothing here executes a comparison.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from neuralls.composition.comparison.config_assembler import resolve_dataset_dir
from neuralls.composition.comparison.models import (
    ComparisonGroup,
    ComparisonGroupings,
    GroupKey,
    GroupMatrixIdentity,
    MultiMatrixDataset,
    SingleMatrixDataset,
)
from neuralls.composition.comparison.pinned_split import resolve_matrix_index
from neuralls.platform.config.models.experiments import CaseConfig, ComparisonRegistryEntry
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.dataset_readers import list_available_matrix_indices
from neuralls.shared.digest import Digest
from neuralls.shared.types import ComparisonRhsSourceKind, DatasetId, GroupId, MatrixId

__all__ = [
    "discover_comparison_groupings",
    "group_key_by_matrix",
    "group_key_by_matrix_and_rhs_kind",
    "validate_group_specs_consistent",
    "validate_one_matrix_shape_per_case",
]

type GroupKeyFn = Callable[[ComparisonRegistryEntry], GroupKey]


def group_key_by_matrix(entry: ComparisonRegistryEntry) -> GroupKey:
    """Partition key for execution-time setup-sharing and the pooled report.

    `rhs_source.kind` plays no part: a preconditioner's `setup()` never sees
    the RHS, so whether its build can be reused across entries is a question
    about `matrix_dataset` alone.
    """
    return GroupKey(dataset_id=DatasetId(entry.matrix_dataset), rhs_kind=None)


def group_key_by_matrix_and_rhs_kind(entry: ComparisonRegistryEntry) -> GroupKey:
    """Partition key for the headline report: one matrix family, one RHS distribution."""
    return GroupKey(
        dataset_id=DatasetId(entry.matrix_dataset),
        rhs_kind=ComparisonRhsSourceKind(entry.rhs_source.kind),
    )


def _matrix_count(cache: dict[DatasetId, int], dataset_id: DatasetId, dataset_dir: Path) -> int:
    """Number of matrix samples in a dataset, read from disk once per dataset."""
    if dataset_id not in cache:
        cache[dataset_id] = len(list_available_matrix_indices(dataset_dir))
    return cache[dataset_id]


def _resolve_identity(
    members: Sequence[ComparisonRegistryEntry],
    dataset_id: DatasetId,
    dataset_dir: Path,
    matrix_count: int,
) -> GroupMatrixIdentity:
    if matrix_count <= 1:
        return SingleMatrixDataset(dataset_id=dataset_id)
    return MultiMatrixDataset(
        dataset_id=dataset_id,
        resolved_indices=frozenset(
            MatrixId(
                resolve_matrix_index(
                    dataset_dir,
                    matrix_index=member.matrix_index,
                    test_position=member.test_position,
                    rhs_source_kind=ComparisonRhsSourceKind(member.rhs_source.kind),
                )
            )
            for member in members
        ),
    )


def _discover_groups(
    all_entries: Sequence[ComparisonRegistryEntry],
    dataset_dirs: Mapping[DatasetId, Path],
    matrix_counts: dict[DatasetId, int],
    *,
    key_fn: GroupKeyFn,
) -> Mapping[GroupId, ComparisonGroup]:
    """Group every entry by `key_fn`, preserving config order within each group."""
    members_by_key: dict[GroupKey, list[ComparisonRegistryEntry]] = {}
    for entry in all_entries:
        members_by_key.setdefault(key_fn(entry), []).append(entry)

    groups: dict[GroupId, ComparisonGroup] = {}
    for key, members in members_by_key.items():
        dataset_dir = dataset_dirs[key.dataset_id]
        identity = _resolve_identity(
            members,
            key.dataset_id,
            dataset_dir,
            _matrix_count(matrix_counts, key.dataset_id, dataset_dir),
        )
        kind_counts = Counter(ComparisonRhsSourceKind(member.rhs_source.kind) for member in members)
        group = ComparisonGroup(
            key=key,
            matrix_identity=identity,
            entry_ids=tuple(member.id for member in members),
            rhs_kind_counts=tuple(kind_counts.items()),
        )
        groups[group.group_id] = group
    return groups


def _dataset_dirs(
    all_entries: Sequence[ComparisonRegistryEntry],
    master_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> dict[DatasetId, Path]:
    return {
        DatasetId(dataset_id): resolve_dataset_dir(master_cfg, config_dir, dataset_id, settings)
        for dataset_id in dict.fromkeys(entry.matrix_dataset for entry in all_entries)
    }


def discover_comparison_groupings(
    all_entries: Sequence[ComparisonRegistryEntry],
    master_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
    *,
    by_matrix_key: GroupKeyFn = group_key_by_matrix,
    by_matrix_and_rhs_kind_key: GroupKeyFn = group_key_by_matrix_and_rhs_kind,
) -> ComparisonGroupings:
    """Discover both grouping granularities one batch needs, reading each dataset once."""
    dataset_dirs = _dataset_dirs(all_entries, master_cfg, config_dir, settings)
    matrix_counts: dict[DatasetId, int] = {}
    return ComparisonGroupings(
        by_matrix=_discover_groups(all_entries, dataset_dirs, matrix_counts, key_fn=by_matrix_key),
        by_matrix_and_rhs_kind=_discover_groups(
            all_entries, dataset_dirs, matrix_counts, key_fn=by_matrix_and_rhs_kind_key
        ),
    )


def validate_one_matrix_shape_per_case(
    all_entries: Sequence[ComparisonRegistryEntry],
    master_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> None:
    """Reject a case that mixes single-matrix and multi-matrix dataset references.

    A case is one coherent experimental question — a multi-matrix case and a
    single-matrix one answer genuinely different questions and produce
    differently-shaped plot evidence — so mixing the two in one case file is
    rejected outright rather than left to authoring discipline.
    """
    matrix_counts: dict[DatasetId, int] = {}
    is_multi_by_dataset = {
        dataset_id: _matrix_count(matrix_counts, dataset_id, dataset_dir) > 1
        for dataset_id, dataset_dir in _dataset_dirs(
            all_entries, master_cfg, config_dir, settings
        ).items()
    }
    if len(set(is_multi_by_dataset.values())) <= 1:
        return
    single = sorted(name for name, is_multi in is_multi_by_dataset.items() if not is_multi)
    multi = sorted(name for name, is_multi in is_multi_by_dataset.items() if is_multi)
    raise ValueError(
        "Case config mixes single-matrix and multi-matrix dataset references "
        f"(single-matrix: {single}; multi-matrix: {multi}). A case must be one "
        "matrix shape — split this into separate case files."
    )


def validate_group_specs_consistent(
    groups: Mapping[GroupId, ComparisonGroup],
    identities_by_entry_id: Mapping[str, Mapping[str, Digest]],
) -> None:
    """Reject a group whose members use one preconditioner name for different preconditioners.

    Setup sharing and scaling-law aggregation are both keyed on the
    preconditioner name, so two members that reuse a name for different
    preconditioners would silently serve or average unrelated builds. "Different"
    is decided by identity (config digest plus checkpoint *content* digest, see
    `preconditioner_identities`), not by name or by `==`: a neural name only
    references a model, and the same model resolved to two temp paths is still
    one preconditioner. Members absent from `identities_by_entry_id` (cache hits)
    are skipped — they run nothing this call.

    Raises:
        ValueError: Naming the group, the preconditioner, and both entries.
    """
    for group in groups.values():
        first_seen: dict[str, tuple[str, Digest]] = {}
        for entry_id in group.entry_ids:
            for name, identity in identities_by_entry_id.get(entry_id, {}).items():
                previous = first_seen.setdefault(name, (entry_id, identity))
                if previous[1] != identity:
                    raise ValueError(
                        f"Group '{group.group_id}': preconditioner '{name}' is configured "
                        f"differently in entries '{previous[0]}' and '{entry_id}' (different "
                        "settings or a different checkpoint behind the same name). Members of "
                        "one group must share a preconditioner definition per name — rename it "
                        "or move one entry to a different dataset/RHS group."
                    )
