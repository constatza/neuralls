"""Prepare every system of a case: validate, resolve and describe."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from loguru import logger
from mlflow.exceptions import MlflowException

from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.composition.assignments.comparison_prepared import (
    PendingComparison,
    PreparedComparisonBatch,
    PreparedComparisonExecution,
    partition_pending,
)
from neuralls.composition.assignments.comparison_resolution import (
    ComparisonResolutionContext,
    ResolvedComparisonSpecs,
    existing_assignment_ids,
    neural_specs_from_assignments,
    require_rhs_source_kind,
    resolve_comparison_specs,
)
from neuralls.composition.assignments.comparison_topology import ComparisonTopology
from neuralls.composition.assignments.presentation_preflight import (
    compose_comparison_title,
    prepare_comparison_presentation,
)
from neuralls.composition.comparison.config_assembler import (
    resolve_comparison_config,
    resolve_dataset_dir,
)
from neuralls.composition.comparison.input_gate import require_current_inputs
from neuralls.composition.comparison.method_data_origin import (
    CheckpointBased,
    TrainingDataUnavailable,
    resolve_method_data_origin,
)
from neuralls.composition.comparison.models import (
    ComparisonEntryFailed,
    ComparisonOutcome,
    PlannedSystem,
    UndescribedComparison,
)
from neuralls.composition.comparison.plan import plan_comparisons
from neuralls.composition.comparison.result_keys import validate_unique_preconditioner_keys
from neuralls.composition.comparison.source_handlers import (
    ComparisonSourceSpec,
    validate_comparison_source,
)
from neuralls.composition.comparison.spec_consistency import validate_dataset_specs_consistent
from neuralls.composition.identity.comparison import (
    comparison_identity,
    preconditioner_identities,
)
from neuralls.composition.identity.training import assignment_dataset_dir
from neuralls.composition.tracking.run_specs import (
    comparison_run,
)
from neuralls.composition.tracking.services import TrackingServices
from neuralls.domain.benchmarking.selection import (
    ReferenceSelection,
    resolve_reference_selection,
    validate_reference_preconditioner,
)
from neuralls.domain.presentation.run_names import Untimed
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
)
from neuralls.domain.solver.models.cost import Known, Unavailable
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    ComparisonRegistryEntry,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.dataset_digest import current_dataset_digest
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.platform.tracking.run_duration import run_wall_seconds_from_run
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.shared.digest import Digest


def _undescribed_failure(entry: ComparisonRegistryEntry, exc: Exception) -> ComparisonOutcome:
    """The outcome of a comparison that failed before its methods and dataset were resolved."""
    logger.error("A comparison failed before it could be described: {}", exc)
    return ComparisonEntryFailed(
        comparison_id=entry.id, run=UndescribedComparison(), reason=str(exc)
    )


def matrix_scope(
    planned: PlannedSystem, context: ComparisonResolutionContext
) -> ComparisonScopePresentation:
    """The presentation of the matrix dataset a comparison runs on, from its manifest."""
    case_cfg = (
        context.master_cfg
        if context.master_cfg is not None
        else load_validated_case_config(context.case_config_path, context.settings)[0]
    )
    dataset_dir = resolve_dataset_dir(
        case_cfg,
        context.case_config_path.resolve().parent,
        planned.entry.matrix_dataset,
        context.settings,
    )
    return ComparisonScopePresentation(context.load_dataset(dataset_dir))


def prepare_comparison_entry(
    cfg: ComparisonConfig,
    planned: PlannedSystem,
    context: ComparisonResolutionContext,
) -> PendingComparison:
    """Validate, resolve and describe one entry. Opens no MLflow run.

    Method-child reuse is checked only after the complete batch is prepared,
    because a legacy per-system identity cannot prove stored method provenance.

    Artifact leases come from `context.artifact_leases` when set (the sweep
    case: one manager, owned and closed by the caller, so each MLflow artifact
    is downloaded once however many entries reference it and every download
    lives until the sweep ends, failures included). Without it,
    the entry owns a private manager: every terminal outcome (validation
    failure or resolution failure) closes it before returning,
    and a `PreparedComparisonExecution` instead detaches that lifetime for the
    caller to keep open through execution.

    Args:
        cfg: Fully resolved ComparisonConfig with injected data paths and preconditioners.
        planned: The planned system, whose entry provides display name and method path.
        context: Shared resolution context (topology, case config, settings).

    Returns:
        A failure `ComparisonOutcome`, or a `PreparedComparisonExecution` ready for execution.
    """
    entry = planned.entry
    try:
        validate_comparison_sources(cfg)
    except (FileNotFoundError, ValueError) as exc:
        return _undescribed_failure(entry, exc)
    configured_reference = _configured_reference(cfg)

    with contextlib.ExitStack() as local_stack:
        artifact_leases = context.artifact_leases or local_stack.enter_context(
            context.tracking.lease_manager(context.topology.model_store_tracking_uri)
        )
        try:
            resolved = resolve_comparison_specs(cfg, context, artifact_leases)
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            return _undescribed_failure(entry, exc)
        logger.info(
            "Comparison '{}': resolved {} preconditioners with {} warnings.",
            entry.id,
            len(resolved.specs),
            len(resolved.warnings),
        )
        training_client = context.tracking.client(context.topology.model_store_tracking_uri)
        run_records = context.training_run_records

        def run_tags(run_id: str) -> dict[str, str]:
            if run_id not in run_records:
                try:
                    run_records[run_id] = training_client.get_run(run_id)
                except (MlflowException, OSError) as error:
                    run_records[run_id] = error
            record = run_records[run_id]
            if isinstance(record, MlflowException | OSError):
                raise record
            return record.data.tags

        origins_by_method = {
            spec.name: resolve_method_data_origin(
                spec,
                run_tags=run_tags,
                training_dataset_dir=lambda path: assignment_dataset_dir(path, context.settings),
                dataset_digest=current_dataset_digest,
            )
            for spec in resolved.specs
        }
        if len(origins_by_method) != len(resolved.specs):
            raise ValueError("resolved method names must be unique")
        training_run_costs: dict[str, Known | Unavailable] = {}
        for run_id, record in run_records.items():
            if isinstance(record, MlflowException):
                training_run_costs[run_id] = Unavailable(
                    f"run {run_id} cannot be read: {record.message}"
                )
            elif isinstance(record, OSError):
                training_run_costs[run_id] = Unavailable(f"run {run_id} cannot be read: {record}")
            else:
                training_run_costs[run_id] = run_wall_seconds_from_run(record, run_id)
        for spec in resolved.specs:
            origin = origins_by_method[spec.name]
            if isinstance(origin, CheckpointBased) and isinstance(
                origin.training_data, TrainingDataUnavailable
            ):
                logger.warning(
                    "Method '{}': training data provenance unavailable: {}",
                    spec.name,
                    origin.training_data.reason,
                )
        reference_selection = _reference_selection(configured_reference, resolved)

        scope = matrix_scope(planned, context)
        presentation = prepare_comparison_presentation(
            resolved.specs, origins_by_method, scope, context.load_dataset
        )
        title = compose_comparison_title(entry.display_name, scope.dataset)
        run = comparison_run(planned, title, scope.dataset, cfg.notation_policy, Untimed())

        logger.info(
            "{}: building comparison identity, including matrix/dataset content.",
            run_name_text(run),
        )
        identity = comparison_identity(cfg, resolved.specs, resolved.checkpoint_digests)
        logger.info("{}: built identity {}.", run_name_text(run), identity.key)
        # Needs real execution: detach the lease manager's cleanup from this
        # function's scope-exit and hand it to the caller instead.
        return PreparedComparisonExecution(
            cfg=cfg,
            planned=planned,
            topology=context.topology,
            cleanup=local_stack.pop_all(),
            resolved=resolved,
            origins_by_method=origins_by_method,
            training_run_costs=training_run_costs,
            identity=identity,
            settings=context.settings,
            presentation=presentation,
            title=title,
            reference_selection=reference_selection,
        )


def _configured_reference(cfg: ComparisonConfig) -> str | None:
    """The configured reference preconditioner name, validated against the configured methods."""
    configured = cfg.general.reference_preconditioner
    reference = configured if isinstance(configured, str) else None
    validate_reference_preconditioner(reference, tuple(spec.name for spec in cfg.preconditioners))
    return reference


def _reference_selection(
    configured_reference: str | None, resolved: ResolvedComparisonSpecs
) -> ReferenceSelection:
    """The reference state after resolution, recording a reference that was skipped as unresolved."""
    if configured_reference is None:
        skipped: dict[str, str] = {}
    else:
        first_warning = next(
            (warning for warning in resolved.warnings if configured_reference in warning), None
        )
        skipped = {} if first_warning is None else {configured_reference: first_warning}
    return resolve_reference_selection(
        configured_reference,
        tuple(spec.name for spec in resolved.specs),
        skipped=skipped,
    )


def validate_comparison_sources(cfg: ComparisonConfig) -> None:
    """Validate only the concrete input artifacts required by this workflow mode.

    Dispatches through the same `ComparisonSourceHandler` registry that
    resolves the inputs, so preflight validation and resolution can never
    disagree about which kinds are supported or which files a kind reads.
    """
    validate_comparison_source(
        ComparisonSourceSpec(
            matrix_path=Path(cfg.general.data.matrix_path),
            rhs_source_kind=require_rhs_source_kind(cfg),
            rhs_source_params=cfg.general.data.rhs_source_params or {},
        )
    )


def prepare_comparison_batch(
    master_cfg: CaseConfig,
    topology: ComparisonTopology,
    case_config_path: Path,
    settings: NeurallsSettings,
    tracking: TrackingServices,
    stack: contextlib.ExitStack,
) -> PreparedComparisonBatch:
    """Plan, resolve and presentation-validate every system of the case.

    Opens no MLflow run. `plan_comparisons` turns every `[[comparisons]]` and
    `[[comparison_sweeps]]` entry into concrete systems first, so each is
    resolved identically. Systems needing execution have
    their cleanup registered onto `stack`, which the caller must keep open
    until after execution.

    A scientific presentation error (a dataset without provenance, a neural method
    without a display name, two indistinguishable methods) propagates out of this function,
    so no MLflow run is opened and no method executes for a case that cannot be described.

    Returns:
        The prepared batch: one outcome-or-prepared-execution per planned system, in plan order.
    """
    config_dir = case_config_path.resolve().parent
    split_records = require_current_inputs(
        master_cfg,
        config_dir,
        settings,
        [
            entry.matrix_dataset
            for entry in (*master_cfg.comparisons, *master_cfg.comparison_sweeps)
        ],
    )
    context = ComparisonResolutionContext(
        topology=topology,
        case_config_path=case_config_path,
        settings=settings,
        load_dataset=load_dataset_presentation,
        tracking=tracking,
        master_cfg=master_cfg,
        artifact_leases=stack.enter_context(
            tracking.lease_manager(topology.model_store_tracking_uri)
        ),
    )
    pending: list[PendingComparison] = []
    planned_systems = plan_comparisons(master_cfg, config_dir, settings)
    for planned in planned_systems:
        cfg = _with_assignment_preconditioners(
            resolve_comparison_config(master_cfg, config_dir, planned, settings),
            planned,
            master_cfg=master_cfg,
            config_dir=config_dir,
            settings=settings,
            tracking=tracking,
            tracking_uri=topology.model_store_tracking_uri,
        )
        prepared = prepare_comparison_entry(cfg, planned, context)
        if isinstance(prepared, PreparedComparisonExecution):
            stack.enter_context(prepared.cleanup)
        pending.append(prepared)
    validate_dataset_specs_consistent(planned_systems, _identities_by_entry_id(pending))
    return PreparedComparisonBatch(
        items=tuple(pending), planned=planned_systems, split_records=split_records
    )


def _with_assignment_preconditioners(
    cfg: ComparisonConfig,
    planned: PlannedSystem,
    *,
    master_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
    tracking: TrackingServices,
    tracking_uri: str,
) -> ComparisonConfig:
    """Append the auto-generated preconditioners of the assignments no explicit one claims."""
    entry = planned.entry
    assignment_entries: list[AssignmentEntry] = (
        [e for e in master_cfg.assignments if e.id in entry.assignments]
        if entry.assignments
        else list(master_cfg.assignments)
    )
    auto_specs = neural_specs_from_assignments(
        assignment_entries,
        existing_assignment_ids(cfg.preconditioners),
        tracking=tracking,
        tracking_uri=tracking_uri,
        cfg=master_cfg,
        config_dir=config_dir,
        settings=settings,
    )
    if auto_specs:
        cfg = replace(cfg, preconditioners=cfg.preconditioners + tuple(auto_specs))
    validate_unique_preconditioner_keys(cfg.preconditioners)
    return cfg


def _identities_by_entry_id(
    pending: Sequence[PendingComparison],
) -> dict[str, dict[str, Digest]]:
    """Identity of every preconditioner of every entry that will execute, by name."""
    return {
        p.entry.id: preconditioner_identities(p.resolved.specs, p.resolved.checkpoint_digests)
        for p in partition_pending(pending).executions
    }
