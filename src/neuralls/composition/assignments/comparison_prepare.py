"""Prepare every system of a case: validate, resolve, reuse-check and describe."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from loguru import logger
from mlflow.tracking import MlflowClient

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
from neuralls.composition.comparison.models import (
    ComparisonEntryFailed,
    ComparisonOutcome,
    ComparisonReused,
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
from neuralls.composition.identity.gate import gate_reuse
from neuralls.composition.tracking.run_specs import (
    comparison_run,
)
from neuralls.domain.benchmarking.selection import (
    ReferenceSelection,
    resolve_reference_selection,
    validate_reference_preconditioner,
)
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.run_names import NamedRun, Untimed
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
)
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    ComparisonRegistryEntry,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.platform.tracking.artifact_access import (
    MlflowArtifactLeaseManager,
)
from neuralls.platform.tracking.mlflow_store import (
    MlflowIdentityStore,
)
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
    *,
    force: bool = False,
) -> PendingComparison:
    """Validate, resolve, and reuse-check one entry. Opens no MLflow run.

    Preconditioners are resolved before deciding whether to run at all: if a
    FINISHED comparison run already exists carrying this comparison's derived
    identity (config, data, preconditioners and checkpoint content), execution is skipped and that
    success is reported directly (unless force=True) — no MLflow run is ever
    opened for a cache hit.

    Artifact leases come from `context.artifact_leases` when set (the sweep
    case: one manager, owned and closed by the caller, so each MLflow artifact
    is downloaded once however many entries reference it and every download
    lives until the sweep ends, cache hits and failures included). Without it,
    the entry owns a private manager: every terminal outcome (validation
    failure, resolution failure, or a cache hit) closes it before returning,
    and a `PreparedComparisonExecution` instead detaches that lifetime for the
    caller to keep open through execution.

    Args:
        cfg: Fully resolved ComparisonConfig with injected data paths and preconditioners.
        planned: The planned system, whose entry provides display name and method path.
        context: Shared resolution context (topology, case config, settings).
        force: Treat this entry as needing execution even if a matching comparison
            already completed successfully.

    Returns:
        A cache-hit/failure `ComparisonOutcome`, or a `PreparedComparisonExecution`
        ready for execution.
    """
    entry = planned.entry
    logger.info("Preparing comparison '{}': validating input sources.", entry.id)
    try:
        validate_comparison_sources(cfg)
    except (FileNotFoundError, ValueError) as exc:
        return _undescribed_failure(entry, exc)
    logger.info("Comparison '{}': input-source validation complete.", entry.id)
    configured_reference = _configured_reference(cfg)

    with contextlib.ExitStack() as local_stack:
        artifact_leases = context.artifact_leases or local_stack.enter_context(
            MlflowArtifactLeaseManager(
                client=MlflowClient(tracking_uri=context.topology.model_store_tracking_uri)
            )
        )
        logger.info("Comparison '{}': resolving preconditioner checkpoints.", entry.id)
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
        reference_selection = _reference_selection(configured_reference, resolved)

        scope = matrix_scope(planned, context)
        presentation = prepare_comparison_presentation(resolved.specs, scope, context.load_dataset)
        title = compose_comparison_title(entry.display_name, scope.dataset)
        run = comparison_run(planned, title, scope.dataset, cfg.notation_policy, Untimed())

        logger.info(
            "{}: building comparison identity, including matrix/dataset content.",
            run_name_text(run),
        )
        identity = comparison_identity(cfg, resolved.specs, resolved.checkpoint_digests)
        logger.info("{}: built identity {}.", run_name_text(run), identity.key)
        if _is_reusable(context, identity, run, force=force):
            return ComparisonReused(comparison_id=entry.id, run=run)

        # Needs real execution: detach the lease manager's cleanup from this
        # function's scope-exit and hand it to the caller instead.
        return PreparedComparisonExecution(
            cfg=cfg,
            planned=planned,
            topology=context.topology,
            cleanup=local_stack.pop_all(),
            resolved=resolved,
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


def _is_reusable(
    context: ComparisonResolutionContext, identity: StageIdentity, run: NamedRun, *, force: bool
) -> bool:
    """Whether a FINISHED run with this identity already exists and execution can be skipped."""
    label = run_name_text(run)
    if not context.reuse_comparisons:
        logger.info("{}: per-system reuse is disabled for batch execution.", label)
        return False
    logger.info("{}: checking MLflow for a reusable run.", label)
    reused = gate_reuse(
        MlflowIdentityStore(
            tracking_uri=context.topology.tracking_uri,
            experiment=context.topology.experiment_name,
        ),
        identity,
        force=force,
        label=label,
    )
    if reused is None:
        logger.info("{}: no reusable run selected; execution is required.", label)
        return False
    logger.info("{}: reuse lookup selected run {}.", label, reused.run_id)
    return True


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
    mlflow_client: MlflowClient,
    stack: contextlib.ExitStack,
    *,
    force: bool,
    reuse_comparisons: bool = True,
) -> PreparedComparisonBatch:
    """Plan, resolve, reuse-check and presentation-validate every system of the case.

    Opens no MLflow run. `plan_comparisons` turns every `[[comparisons]]` and
    `[[comparison_sweeps]]` entry into concrete systems first, so each is
    resolved and reuse-checked identically. Systems needing execution have
    their cleanup registered onto `stack`, which the caller must keep open
    until after execution.

    A scientific presentation error (a dataset without provenance, a neural method
    without a display name, two indistinguishable methods) propagates out of this function,
    so no MLflow run is opened and no method executes for a case that cannot be described.

    Returns:
        The prepared batch: one outcome-or-prepared-execution per planned system, in plan order.
    """
    config_dir = case_config_path.resolve().parent
    require_current_inputs(
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
        master_cfg=master_cfg,
        artifact_leases=stack.enter_context(
            MlflowArtifactLeaseManager(
                client=MlflowClient(tracking_uri=topology.model_store_tracking_uri)
            )
        ),
        reuse_comparisons=reuse_comparisons,
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
            mlflow_client=mlflow_client,
        )
        prepared = prepare_comparison_entry(cfg, planned, context, force=force)
        if isinstance(prepared, PreparedComparisonExecution):
            stack.enter_context(prepared.cleanup)
        pending.append(prepared)
    validate_dataset_specs_consistent(planned_systems, _identities_by_entry_id(pending))
    return PreparedComparisonBatch(items=tuple(pending), planned=planned_systems)


def _with_assignment_preconditioners(
    cfg: ComparisonConfig,
    planned: PlannedSystem,
    *,
    master_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
    mlflow_client: MlflowClient,
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
        client=mlflow_client,
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
