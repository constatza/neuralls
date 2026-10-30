"""Group executable systems into the method -> matrix -> RHS schedule and load their inputs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from neuralls.composition.assignments.comparison_prepared import (
    PendingComparison,
    PreparedComparisonBatch,
    PreparedComparisonExecution,
    partition_pending,
)
from neuralls.composition.assignments.comparison_resolution import (
    require_rhs_source_kind,
)
from neuralls.composition.assignments.comparison_topology import ComparisonTopology
from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison._linear_system import prepare_resolved_linear_system
from neuralls.composition.comparison._preconditioner_setup import (
    matrix_key,
)
from neuralls.composition.comparison.models import (
    LinearSystem,
    MatrixWorkload,
    MatrixWorkloadMember,
    ResolvedMethod,
    canonical_method_config,
)
from neuralls.composition.comparison.recipes import system_tag
from neuralls.composition.identity.comparison import (
    preconditioner_identities,
)
from neuralls.composition.tracking.services import TrackingServices
from neuralls.platform.tracking.mlflow_store import (
    find_reusable_benchmark_method,
)
from neuralls.platform.tracking.session_artifacts import SESSION_PHASE
from neuralls.shared.digest import Digest, canonical_digest
from neuralls.shared.types import (
    DatasetId,
    MatrixId,
)

type MatrixKey = tuple[DatasetId, MatrixId]
type MatrixGroup = tuple[MatrixKey, tuple[PreparedComparisonExecution, ...]]
"""One matrix and its executions, in RHS order."""
type MethodSchedule = tuple[tuple[ResolvedMethod, tuple[MatrixGroup, ...]], ...]
"""Executable systems grouped as method -> matrix -> ordered RHS members."""


def _matrix_of(prepared: PreparedComparisonExecution) -> MatrixKey:
    return DatasetId(prepared.entry.matrix_dataset), matrix_key(prepared.planned)


def method_major_schedule(pending: Sequence[PendingComparison]) -> MethodSchedule:
    """Group executable systems as method -> matrix -> ordered RHS members."""
    methods: dict[tuple[str, Digest], ResolvedMethod] = {}
    identities_by_name: dict[str, Digest] = {}
    semantics_by_name: dict[str, tuple[object, object]] = {}
    expected_method_names: frozenset[str] | None = None
    grouped: dict[
        tuple[str, Digest],
        dict[tuple[DatasetId, MatrixId], list[PreparedComparisonExecution]],
    ] = {}
    for item in partition_pending(pending).executions:
        method_names = frozenset(config.name for config in item.resolved.specs)
        if expected_method_names is None:
            expected_method_names = method_names
        elif method_names != expected_method_names:
            raise ValueError(
                "comparison entries have different method membership; "
                "method-major workloads require the same methods for every planned RHS"
            )
        identities = preconditioner_identities(
            item.resolved.specs, item.resolved.checkpoint_digests
        )
        for config in item.resolved.specs:
            identity = identities[config.name]
            previous_identity = identities_by_name.setdefault(config.name, identity)
            if previous_identity != identity:
                raise ValueError(
                    f"method {config.name!r} has different configuration identities "
                    "within one comparison workload"
                )
            semantics = (item.cfg.general.data.normalize_system, item.cfg.general.params)
            previous_semantics = semantics_by_name.setdefault(config.name, semantics)
            if previous_semantics != semantics:
                raise ValueError(
                    f"method {config.name!r} has incompatible matrix/solver semantics "
                    "within one comparison workload"
                )
            method_key = config.name, identities[config.name]
            methods.setdefault(method_key, ResolvedMethod(config, identity_digest=identity))
            grouped.setdefault(method_key, {}).setdefault(_matrix_of(item), []).append(item)
    return tuple(
        (
            method,
            tuple((matrix, tuple(systems)) for matrix, systems in grouped[key].items()),
        )
        for key, method in methods.items()
    )


def _load_linear_system(prepared: PreparedComparisonExecution) -> LinearSystem:
    """Resolve and prepare the one linear system a prepared execution solves."""
    data = prepared.cfg.general.data
    resolved = resolve_comparison_input(
        matrix_path=Path(data.matrix_path),
        matrix_dataset_id=prepared.entry.matrix_dataset,
        matrix_index=data.matrix_index,
        matrix_provenance=prepared.planned.provenance,
        require_non_residual_rhs=data.require_non_residual_rhs,
        seed=data.selection_seed,
        rhs_source_kind=require_rhs_source_kind(prepared.cfg),
        rhs_source_params=data.rhs_source_params,
    )
    return prepare_resolved_linear_system(resolved, data.normalize_system)


def _load_one_matrix_workload(
    dataset: DatasetId,
    matrix_id: MatrixId,
    systems: tuple[PreparedComparisonExecution, ...],
) -> MatrixWorkload:
    """Build one bundle in a helper scope so producer temporaries die before the next load."""
    first_prepared = systems[0]
    first_data = first_prepared.cfg.general.data
    first_system = _load_linear_system(first_prepared)
    members = [
        MatrixWorkloadMember(
            tag=system_tag(first_prepared.planned),
            rhs=first_system.rhs.numpy(),
        )
    ]
    for prepared in systems[1:]:
        members.append(
            MatrixWorkloadMember(
                tag=system_tag(prepared.planned),
                rhs=_load_linear_system(prepared).rhs.numpy(),
            )
        )
    return MatrixWorkload(
        dataset=dataset,
        matrix_id=matrix_id,
        matrix=first_system.matrix.numpy(),
        matrix_path=Path(first_data.matrix_path),
        matrix_index=first_data.matrix_index,
        members=tuple(members),
    )


def load_matrix_workloads(matrices: Sequence[MatrixGroup]) -> Iterator[MatrixWorkload]:
    """Yield one method matrix/RHS bundle at a time before online timing begins."""
    for (dataset, matrix_id), systems in matrices:
        yield _load_one_matrix_workload(dataset, matrix_id, systems)


def has_complete_reusable_session(
    batch: PreparedComparisonBatch,
    topology: ComparisonTopology,
    tracking: TrackingServices,
    *,
    force: bool,
) -> bool:
    """Require current method children under one finished comparison session parent."""
    if force:
        return False
    schedule = method_major_schedule(batch.items)
    if not schedule:
        return False
    client = tracking.client(topology.tracking_uri)
    parent_ids: set[str] = set()
    for method, matrices in schedule:
        system_ids = tuple(str(system.entry.id) for _, systems in matrices for system in systems)
        system_identities = tuple(
            system.identity.key for _, systems in matrices for system in systems
        )
        reused = find_reusable_benchmark_method(
            tracking_uri=topology.tracking_uri,
            experiment=topology.experiment_name,
            method=str(method.name),
            config_digest=method.identity_digest or canonical_digest(method.config),
            method_config=canonical_method_config(method.config),
            system_ids=system_ids,
            system_identities=system_identities,
        )
        if reused is None or reused.run_id is None:
            return False
        parent_id = client.get_run(reused.run_id).data.tags.get("mlflow.parentRunId")
        if parent_id is None:
            return False
        parent_ids.add(parent_id)
    if len(parent_ids) != 1:
        return False
    parent_id = parent_ids.pop()
    parent = client.get_run(parent_id)
    return parent.info.status == "FINISHED" and parent.data.tags.get("phase") == SESSION_PHASE
