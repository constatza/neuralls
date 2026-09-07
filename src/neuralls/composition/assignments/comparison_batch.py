"""Backend helpers for comparison workflows."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
from dlkit.infrastructure.config.job_config import FitJobConfig
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.composition.assignments.job_loader import load_experiment_job
from neuralls.composition.assignments.model_resolution import (
    AssignmentModelContext,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison.comparison_run import compare_preconditioners
from neuralls.composition.comparison.config_assembler import resolve_comparison_config
from neuralls.composition.comparison.models import ComparisonOutcome, ComparisonParams
from neuralls.composition.tracking.run_specs import (
    ComparisonRunTags,
    build_child_comparison_tags,
    build_comparison_run_spec,
    build_session_run_spec,
)
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.comparison import (
    ComparisonConfig,
    DatasetRhsSourceModel,
    RawLhsSourceModel,
    RawRhsSourceModel,
)
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    ComparisonRegistryEntry,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    CheckpointRefBearing,
    LoggedModelRefConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
    PreconditionerType,
)
from neuralls.platform.config.registry import get_assignment_binding, resolve_assignment_binding
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.artifacts import (
    coerce_comparison_result_payload,
    write_comparison_artifacts,
)
from neuralls.platform.reporting.comparison_inputs import (
    ComparisonInputArtifacts,
    stage_comparison_inputs,
)
from neuralls.platform.storage.validation import (
    validate_comparison_inputs,
    validate_comparison_matrix_input,
    validate_comparison_rhs_input,
)
from neuralls.platform.tracking.artifact_access import (
    ArtifactLeaseManager,
    MlflowArtifactLeaseManager,
)
from neuralls.platform.tracking.comparison_tracking import (
    log_comparison_artifact_uri,
    log_comparison_input_artifacts,
    log_comparison_result_metrics,
    log_comparison_run_params,
    log_linear_system_params,
    log_skipped_preconditioners,
    setup_comparison_tracking,
)
from neuralls.platform.tracking.extra_features import fetch_extra_input_names_for_model
from neuralls.platform.tracking.mlflow import build_workflow_environment
from neuralls.platform.tracking.mlflow_client import (
    find_successful_comparison_run,
    log_comparison_artifacts_to_mlflow,
)
from neuralls.platform.tracking.model_registry import build_registered_model_name
from neuralls.shared.types import ComparisonRhsSourceKind


@dataclass(frozen=True)
class ComparisonTopology:
    """Shared MLflow topology for comparison execution."""

    tracking_uri: str
    artifact_location: str | None
    experiment_name: str
    model_store_tracking_uri: str


def _validate_neural_preconditioner(spec: Any) -> None:
    """Validate strict neural preconditioner requirements for comparisons."""
    if not isinstance(spec, NeuralPreconditionerConfig):
        return
    if spec.model_ref is None:
        raise ValueError(f"Neural preconditioner '{spec.name}' must define model_ref.")
    if spec.checkpoint_path is not None:
        raise ValueError(
            f"Neural preconditioner '{spec.name}' cannot use checkpoint_path/assignment "
            "legacy resolution in comparison configs."
        )


def _resolve_preconditioner(
    spec: Any,
) -> Any:
    """Validate one preconditioner and any assignment-bound references."""
    _validate_neural_preconditioner(spec)
    return spec


def _resolve_neural_preconditioners(
    solver_specs: list[Any],
) -> list[PreconditionerConfig]:
    """Validate neural preconditioners and return unchanged specs."""
    return [_resolve_preconditioner(spec) for spec in solver_specs]


def _needs_model_resolution(specs: tuple[PreconditionerConfig, ...]) -> bool:
    """Return True when any checkpoint-bearing preconditioner needs model_ref lookup.

    Generic over `CheckpointRefBearing` (not `NeuralPreconditionerConfig`-only)
    so a checkpoint-backed `AMGPreconditionerConfig` (POD-2G fitted ahead of
    time via a `FitJobConfig` assignment) triggers resolution the same way a
    neural preconditioner does — one dispatch point, no per-kind branching.
    """
    for spec in specs:
        if not isinstance(spec, CheckpointRefBearing):
            continue
        if any(ref.model_ref is not None for _, ref in spec.checkpoint_refs()):
            return True
    return False


def _referenced_assignment_ids(
    specs: tuple[PreconditionerConfig, ...],
) -> tuple[str, ...]:
    """Return assignment ids explicitly referenced by checkpoint-bearing specs."""
    ids: list[str] = []
    for spec in specs:
        if not isinstance(spec, CheckpointRefBearing):
            continue
        for _, ref in spec.checkpoint_refs():
            if ref.assignment is None or ref.assignment in ids:
                continue
            ids.append(ref.assignment)
    return tuple(ids)


def _build_master_assignment_contexts(
    cfg_path: Path,
    assignment_ids: tuple[str, ...],
    settings: NeurallsSettings,
) -> dict[str, AssignmentModelContext]:
    """Resolve dataset/model identity for all case-config assignments."""
    master_cfg, config_dir = load_validated_case_config(cfg_path, settings)
    contexts: dict[str, AssignmentModelContext] = {}
    for assignment_id in assignment_ids:
        binding = get_assignment_binding(master_cfg, config_dir, assignment_id)
        data_cfg = load_data_config(binding.data_config_path, settings)
        dataset_id = resolve_dataset_identity(
            data_cfg=data_cfg,
            config_path=binding.data_config_path,
        ).name
        contexts[assignment_id] = AssignmentModelContext(
            dataset_alias=dataset_id,
            model_name=build_registered_model_name(assignment_id),
        )
    return contexts


def _load_master_config(
    case_config_path: Path,
    settings: NeurallsSettings,
) -> tuple[CaseConfig, Path]:
    """Load the case config and its resolution root."""
    return load_validated_case_config(case_config_path, settings)


def _existing_assignment_ids(specs: tuple[PreconditionerConfig, ...]) -> set[str]:
    """Return assignment ids already claimed by explicit checkpoint-bearing preconditioners."""
    ids: set[str] = set()
    for spec in specs:
        if not isinstance(spec, CheckpointRefBearing):
            continue
        for _, ref in spec.checkpoint_refs():
            if ref.assignment is not None:
                ids.add(ref.assignment)
    return ids


@dataclass(frozen=True)
class _AssignmentJobContext:
    """One assignment's resolved job settings plus its dataset directory."""

    job: AnyJobConfig
    dataset_dir: Path


def _resolve_assignment_job_context(
    entry: AssignmentEntry,
    *,
    cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> _AssignmentJobContext:
    """Load one assignment's declared job and resolve its dataset directory.

    Both are needed to decide *what kind* of auto-generated preconditioner
    spec an assignment produces (`neural_specs_from_assignments`'s kind
    dispatch) and, for a `fit`-kind job, to populate the resulting
    `PODCoarseningConfig`'s required `dataset_dir`/`rank` fields.
    """
    binding = resolve_assignment_binding(cfg, config_dir, entry)
    job = load_experiment_job(binding.job_config_path, settings)
    data_cfg = load_data_config(binding.data_config_path, settings)
    dataset_id = resolve_dataset_identity(
        data_cfg=data_cfg,
        config_path=binding.data_config_path,
    ).name
    return _AssignmentJobContext(job=job, dataset_dir=settings.processed_dir / dataset_id)


def _neural_spec_from_assignment(
    entry: AssignmentEntry,
    *,
    client: MlflowClient,
) -> NeuralPreconditionerConfig:
    """Build an unresolved neural preconditioner stub for a train/search-kind assignment.

    References the most recent unregistered training run tagged with the
    assignment's id, rather than a registry entry — automatic model
    consumption reads raw MLflow runs, since the registry is reserved for
    deliberate/manual promotion. Also fetches the
    ``neuralls.extra_feature_names`` tag from the training run so that FiLM
    models receive their condition tensor during comparison.
    """
    return NeuralPreconditionerConfig(
        name=entry.effective_display_name,
        type=PreconditionerType.NEURAL,
        assignment=entry.id,
        model_ref=LoggedModelRefConfig(latest=True, tags={"assignment_id": entry.id}),
        extra_input_names=fetch_extra_input_names_for_model(entry.id, client),
    )


def _pod_fit_spec_from_assignment(
    entry: AssignmentEntry,
    *,
    job: FitJobConfig,
    dataset_dir: Path,
) -> AMGPreconditionerConfig:
    """Build an unresolved AMG/POD stub for a `fit`-kind (POD-2G) assignment.

    Mirrors `_neural_spec_from_assignment`'s "unresolved stub referencing the
    most recent tagged run" shape, but for `AMGPreconditionerConfig(coarsening
    =PODCoarseningConfig(...))` instead of `NeuralPreconditionerConfig` — the
    checkpoint-shaped side of the kind dispatch in
    `neural_specs_from_assignments`. `dataset_dir`/`rank` are populated from
    the assignment's own job/dataset so the config stays valid (required
    fields) even before resolution runs; once `resolved_checkpoint_path` is
    set by `resolve_preconditioner_models_with_warnings`, the factory
    reconstructs the fitted basis from the checkpoint and never reads these.
    """
    rank = getattr(job.model, "rank", None)
    if rank is None:
        raise ValueError(
            f"Assignment '{entry.id}' job declares run.type='fit' but its model has no "
            "'rank' hyperparameter — expected a PODCoarseningStrategy-shaped model."
        )
    return AMGPreconditionerConfig(
        name=entry.effective_display_name,
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(
            dataset_dir=dataset_dir,
            rank=rank,
            assignment=entry.id,
            model_ref=LoggedModelRefConfig(latest=True, tags={"assignment_id": entry.id}),
        ),
    )


def neural_specs_from_assignments(
    entries: Sequence[AssignmentEntry],
    claimed_ids: set[str],
    *,
    client: MlflowClient,
    cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> list[PreconditionerConfig]:
    """Generate auto preconditioner stubs from assignment entries.

    Dispatches on each assignment's declared job kind: a `fit`-kind job
    (`FitJobConfig` — e.g. POD-2G basis fitting) yields an unresolved
    `AMGPreconditionerConfig`/`PODCoarseningConfig` stub; any other kind
    (today: `train`/`search`) yields the existing unresolved
    `NeuralPreconditionerConfig` stub. Both stubs reference the most recent
    run tagged with the assignment's id and get resolved to a concrete
    checkpoint later, symmetrically, by
    `resolve_preconditioner_models_with_warnings` (generic over
    `CheckpointRefBearing`). Closed for modification, open for extension: a
    future `Fittable` artifact kind adds one more dispatch branch here, not a
    new isinstance chain elsewhere.

    Args:
        entries: Assignment entries to convert.
        claimed_ids: Assignment ids already covered by explicit preconditioners.
        client: Configured MLflow client used to look up training run tags.
        cfg: The assignments' owning case config, for job/dataset resolution.
        config_dir: Directory `cfg`'s registry paths resolve relative to.
        settings: Runtime settings used to load each assignment's job/dataset config.

    Returns:
        List of auto-generated preconditioner configs, one per unclaimed entry.
    """
    specs: list[PreconditionerConfig] = []
    for entry in entries:
        if entry.id in claimed_ids:
            continue
        context = _resolve_assignment_job_context(
            entry, cfg=cfg, config_dir=config_dir, settings=settings
        )
        if isinstance(context.job, FitJobConfig):
            specs.append(
                _pod_fit_spec_from_assignment(
                    entry, job=context.job, dataset_dir=context.dataset_dir
                )
            )
        else:
            specs.append(_neural_spec_from_assignment(entry, client=client))
    return specs


def _resolve_specs(
    cfg: ComparisonConfig,
    case_config_path: Path,
    model_store_tracking_uri: str,
    settings: NeurallsSettings,
    artifact_leases: ArtifactLeaseManager,
) -> tuple[list[PreconditionerConfig], tuple[str, ...]]:
    """Resolve model_ref preconditioners into concrete checkpoint paths."""
    specs = _resolve_neural_preconditioners(list(cfg.preconditioners))
    if not _needs_model_resolution(cfg.preconditioners):
        return specs, ()

    assignment_ids = _referenced_assignment_ids(cfg.preconditioners)
    assignment_contexts = None
    if assignment_ids:
        assignment_contexts = _build_master_assignment_contexts(
            case_config_path, assignment_ids, settings
        )
    resolution = resolve_preconditioner_models_with_warnings(
        specs=specs,
        tracking_uri=model_store_tracking_uri,
        artifact_leases=artifact_leases,
        dataset_alias=cfg.general.data.dataset_alias,
        assignment_contexts=assignment_contexts,
        skip_unresolved=True,
    )
    return resolution.specs, resolution.warnings


def _compute_checkpoint_dependency_hash(specs: Sequence[PreconditionerConfig]) -> str:
    """Hash identifying which trained checkpoints this comparison depends on.

    Not a hash of any file or of the preconditioners' own config — it's built
    from each dependency's resolved MLflow `run_id` (the training run that
    produced its checkpoint), which is the only stable identity available: the
    checkpoint file itself is re-leased to a fresh temp path on every
    invocation (`MlflowArtifactLeaseManager`), so its local path/mtime can't
    serve as one. Changes whenever any dependency's resolved training run
    changes, so a prior comparison's reuse-check tag no longer matches once
    training reruns it depends on (e.g. via a dataset regeneration cascading
    through `run_assignment_sweep`'s reuse check).
    """
    parts: list[str] = []
    for spec in specs:
        if not isinstance(spec, CheckpointRefBearing):
            continue
        for label, ref in spec.checkpoint_refs():
            identity = ref.resolved_run_id or str(ref.resolved_checkpoint_path)
            parts.append(f"{spec.name}:{label}:{identity}")
    hasher = hashlib.sha1()
    hasher.update("|".join(sorted(parts)).encode())
    return hasher.hexdigest()


def _resolve_comparison_topology(
    case_config_path: Path,
    settings: NeurallsSettings,
) -> ComparisonTopology:
    """Resolve comparison MLflow topology from the case config."""
    master_cfg, _ = _load_master_config(case_config_path, settings)
    runtime = build_workflow_environment(
        tracking_uri=master_cfg.mlflow.tracking_uri,
        artifact_location=master_cfg.mlflow.artifacts_destination,
        default_output_root=settings.output_dir,
        config_path=case_config_path,
    )
    return ComparisonTopology(
        tracking_uri=runtime.tracking_uri,
        artifact_location=runtime.artifact_uri,
        experiment_name=master_cfg.names.comparison,
        model_store_tracking_uri=runtime.tracking_uri,
    )


def _require_rhs_source_kind(cfg: ComparisonConfig) -> ComparisonRhsSourceKind:
    """Return the configured RHS source kind or fail with a focused error."""
    kind = cfg.general.data.rhs_source_kind
    if kind is None:
        raise ValueError("Comparison config must define rhs_source.")
    return kind


def _run_comparison_with_resolved_specs(
    *,
    cfg: ComparisonConfig,
    entry: ComparisonRegistryEntry,
    work_root: Path,
    resolved_specs: list[PreconditionerConfig],
) -> ComparisonResult:
    """Run comparison with already-resolved preconditioner checkpoint paths."""
    rhs_source_kind = _require_rhs_source_kind(cfg)
    resolved_input = resolve_comparison_input(
        matrix_path=Path(cfg.general.data.matrix_path),
        matrix_dataset_id=entry.matrix_dataset,
        matrix_index=cfg.general.data.matrix_index,
        require_non_residual_rhs=cfg.general.data.require_non_residual_rhs,
        seed=cfg.general.data.selection_seed,
        rhs_source_kind=rhs_source_kind,
        rhs_source_params=cfg.general.data.rhs_source_params,
    )
    staged_input_dir = stage_comparison_inputs(
        work_root,
        ComparisonInputArtifacts(
            matrix=resolved_input.matrix,
            rhs=resolved_input.rhs,
            matrix_dataset_id=resolved_input.matrix_dataset_id,
            matrix_index=resolved_input.matrix_index,
            rhs_source_type=resolved_input.rhs_source_type,
            lhs=resolved_input.lhs,
            rhs_dataset_id=resolved_input.rhs_dataset_id,
            rhs_sample_index=resolved_input.rhs_sample_index,
            rhs_kind=resolved_input.rhs_kind,
            rhs_source_kind=resolved_input.rhs_source_kind,
            rhs_source_params=resolved_input.rhs_source_params,
        ),
    )
    raw_result = compare_preconditioners(
        general_params=cfg.general,
        preconditioner_configs=resolved_specs,
        output_root=work_root,
        display_name=entry.effective_display_name,
        resolved_input=resolved_input,
    )
    write_comparison_artifacts(
        result=coerce_comparison_result_payload(raw_result),
        work_root=work_root,
        comparison_config=entry.method,
    )
    log_comparison_input_artifacts(staged_input_dir)
    return raw_result


def _execute_comparison_in_run(
    cfg: ComparisonConfig,
    entry: ComparisonRegistryEntry,
    topology: ComparisonTopology,
    run_name: str,
    comp_tags: ComparisonRunTags,
    checkpoint_dependency_hash: str,
    resolved_specs: list[PreconditionerConfig],
    warnings: tuple[str, ...],
) -> list[ComparisonOutcome]:
    """Open an MLflow run, execute the comparison, upload artifacts, and return outcomes.

    Args:
        cfg: Fully resolved comparison configuration.
        entry: Registry entry for display name, id, and method config path.
        topology: Resolved MLflow topology (tracking URI, artifact location).
        run_name: Display name for the MLflow run.
        comp_tags: Structured tags applied to the MLflow run.
        checkpoint_dependency_hash: Hash of the resolved checkpoints this
            comparison depends on, tagged onto the run for reuse-checking.
        resolved_specs: Already-resolved preconditioner checkpoint paths.
        warnings: Resolution warnings to log alongside the run.

    Returns:
        Single-element list with a successful ComparisonOutcome.
    """
    with mlflow.start_run(
        run_name=run_name,
        nested=True,
        tags={
            **comp_tags.as_mlflow_tags(),
            "checkpoint_dependency_hash": checkpoint_dependency_hash,
        },
    ) as comp_run:
        comp_run_id = comp_run.info.run_id
        log_comparison_artifact_uri()

        with tempfile.TemporaryDirectory() as _tmp:
            work_root = Path(_tmp)
            raw_result = _run_comparison_with_resolved_specs(
                cfg=cfg,
                entry=entry,
                work_root=work_root,
                resolved_specs=resolved_specs,
            )
            log_skipped_preconditioners(warnings)
            log_comparison_artifacts_to_mlflow(
                tracking_uri=topology.tracking_uri,
                run_id=comp_run_id,
                work_root=work_root,
            )
            if isinstance(raw_result, ComparisonResult):
                child_run_tags = {
                    name: build_child_comparison_tags(
                        preconditioner_name=name,
                        comparison_id=entry.id,
                        parent_run_name=run_name,
                    ).as_mlflow_tags()
                    for name in raw_result.results
                }
                log_comparison_result_metrics(raw_result, child_run_tags=child_run_tags)
                log_linear_system_params(raw_result)

        log_comparison_run_params(
            comp_run_id=comp_run_id,
            comparison_id=entry.id,
            comparison_display_name=entry.effective_display_name,
            comparison_config=entry.method.stem if entry.method is not None else entry.id,
        )

    return [_comparison_outcome(entry, success=True, payload=raw_result, warnings=warnings)]


def _comparison_outcome(
    entry: ComparisonRegistryEntry,
    *,
    success: bool,
    error: str | None = None,
    payload: ComparisonResult | None = None,
    warnings: tuple[str, ...] = (),
) -> ComparisonOutcome:
    """Build one comparison's outcome DTO — the one place its identity fields are set."""
    return ComparisonOutcome(
        comparison_id=entry.id,
        comparison_display_name=entry.effective_display_name,
        success=success,
        error=error,
        payload=payload,
        warnings=warnings,
    )


def _resolve_comparison_specs(
    cfg: ComparisonConfig,
    case_config_path: Path,
    topology: ComparisonTopology,
    settings: NeurallsSettings,
    artifact_leases: ArtifactLeaseManager,
) -> tuple[list[PreconditionerConfig], tuple[str, ...], str]:
    """Resolve preconditioner checkpoints and derive this comparison's checkpoint dependency hash.

    Must run inside the same lease-manager scope as execution — leased
    checkpoint files are released when that scope closes.

    Raises:
        ValueError: When all preconditioners fail to resolve.
    """
    resolved_specs, warnings = _resolve_specs(
        cfg,
        case_config_path,
        topology.model_store_tracking_uri,
        settings,
        artifact_leases,
    )
    if not resolved_specs:
        raise ValueError("No runnable preconditioners remain after model resolution.")
    return resolved_specs, warnings, _compute_checkpoint_dependency_hash(resolved_specs)


def _comparison_already_run(
    topology: ComparisonTopology,
    entry: ComparisonRegistryEntry,
    checkpoint_dependency_hash: str,
) -> bool:
    """Whether a FINISHED comparison run already exists for this exact checkpoint dependency hash."""
    return (
        find_successful_comparison_run(
            tracking_uri=topology.tracking_uri,
            mlflow_experiment_name=topology.experiment_name,
            comparison_id=entry.id,
            checkpoint_dependency_hash=checkpoint_dependency_hash,
        )
        is not None
    )


def _run_comparison_from_config(
    cfg: ComparisonConfig,
    entry: ComparisonRegistryEntry,
    topology: ComparisonTopology,
    case_config_path: Path,
    settings: NeurallsSettings,
    *,
    force: bool = False,
) -> list[ComparisonOutcome]:
    """Resolve, reuse-check, and execute one comparison, logging results to MLflow.

    Runs nested under the batch's session parent run (already active), so this
    becomes the "subrun" level between the case-wide parent and per-preconditioner
    leaf runs. Preconditioners are resolved before deciding whether to run at
    all: if a FINISHED comparison run already exists tagged with this
    comparison_id and the exact same checkpoint dependency hash, execution
    is skipped and that success is reported directly (unless force=True).

    Args:
        cfg: Fully resolved ComparisonConfig with injected data paths and preconditioners.
        entry: Registry entry providing display name and method path for artifact logging.
        topology: Resolved MLflow topology for the whole batch (tracking/experiment).
        case_config_path: Case config path for model resolution context.
        settings: Resolved runtime settings.
        force: Rerun even if a matching comparison already completed successfully.

    Returns:
        Single-element list with the outcome of the comparison run.
    """
    try:
        _validate_comparison_sources(cfg)
    except (FileNotFoundError, ValueError) as exc:
        return [_comparison_outcome(entry, success=False, error=str(exc))]

    model_client = MlflowClient(tracking_uri=topology.model_store_tracking_uri)
    try:
        with MlflowArtifactLeaseManager(client=model_client) as artifact_leases:
            resolved_specs, warnings, checkpoint_dependency_hash = _resolve_comparison_specs(
                cfg, case_config_path, topology, settings, artifact_leases
            )

            if not force and _comparison_already_run(topology, entry, checkpoint_dependency_hash):
                logger.info(f"Using existing MLflow comparison run for '{entry.id}'")
                return [_comparison_outcome(entry, success=True)]

            run_name, comp_tags = build_comparison_run_spec(entry=entry, include_timestamp=False)
            return _execute_comparison_in_run(
                cfg=cfg,
                entry=entry,
                topology=topology,
                run_name=run_name,
                comp_tags=comp_tags,
                checkpoint_dependency_hash=checkpoint_dependency_hash,
                resolved_specs=resolved_specs,
                warnings=warnings,
            )
    except (ValueError, RuntimeError, KeyError) as exc:
        logger.error(f"Comparison failed: {exc}")
        return [_comparison_outcome(entry, success=False, error=str(exc))]


def _validate_comparison_sources(cfg: ComparisonConfig) -> None:
    """Validate only the concrete input artifacts required by this workflow mode."""
    matrix_path = Path(cfg.general.data.matrix_path)
    rhs_source_kind = _require_rhs_source_kind(cfg)
    rhs_source_params = cfg.general.data.rhs_source_params or {}
    match rhs_source_kind:
        case ComparisonRhsSourceKind.GAUSSIAN | ComparisonRhsSourceKind.SPARSE:
            validate_comparison_matrix_input(matrix_path)
        case ComparisonRhsSourceKind.DATASET:
            source = DatasetRhsSourceModel.model_validate(
                {"kind": rhs_source_kind, **rhs_source_params}
            )
            validate_comparison_rhs_input(source.path)
        case ComparisonRhsSourceKind.RAW_LHS:
            source = RawLhsSourceModel.model_validate(
                {"kind": rhs_source_kind, **rhs_source_params}
            )
            validate_comparison_inputs(matrix_path=matrix_path, rhs_path=source.path)
        case ComparisonRhsSourceKind.RAW_RHS:
            source = RawRhsSourceModel.model_validate(
                {"kind": rhs_source_kind, **rhs_source_params}
            )
            validate_comparison_inputs(matrix_path=matrix_path, rhs_path=source.path)
        case _:
            raise ValueError(f"Unsupported comparison RHS source: {rhs_source_kind!r}.")


def run_comparison_batch(
    case_config_path: Path,
    params: ComparisonParams,
    settings: NeurallsSettings | None = None,
) -> list[ComparisonOutcome]:
    """Run all configured comparison entries from the case config.

    Opens one MLflow session parent run for the whole batch — named
    ``"<case-name> | <timestamp>"``, mirroring training's session parent — so
    every comparison entry's own run nests underneath it as a "subrun", with
    each entry's per-preconditioner runs nested one level further.

    Args:
        case_config_path: Path to the case config TOML.
        params: Comparison execution parameters (currently just `force`).
        settings: Optional pre-loaded runtime settings.

    Returns:
        List of comparison outcomes, one per [[comparisons]] entry.
    """
    settings = require_settings(settings, case_config_path=case_config_path)
    master_cfg, config_dir = _load_master_config(case_config_path, settings)
    if not master_cfg.comparisons:
        raise ValueError("Case config must define at least one [[comparisons]] entry.")

    topology = _resolve_comparison_topology(case_config_path, settings)
    mlflow_client = MlflowClient(tracking_uri=topology.model_store_tracking_uri)
    setup_comparison_tracking(
        tracking_uri=topology.tracking_uri,
        artifact_location=topology.artifact_location,
        experiment_name=topology.experiment_name,
    )
    session_run_name, session_tags = build_session_run_spec(
        case_config_path=case_config_path.resolve(),
        experiment_name=topology.experiment_name,
        phase="session_comparison",
    )

    outcomes: list[ComparisonOutcome] = []
    with mlflow.start_run(run_name=session_run_name, tags=session_tags.as_mlflow_tags()):
        for entry in master_cfg.comparisons:
            logger.info(f"Comparison [{entry.id}]: {entry.effective_display_name}")
            assignment_entries: list[AssignmentEntry] = (
                [e for e in master_cfg.assignments if e.id in entry.assignments]
                if entry.assignments
                else list(master_cfg.assignments)
            )
            cfg = resolve_comparison_config(master_cfg, config_dir, entry, settings)
            claimed_ids = _existing_assignment_ids(cfg.preconditioners)
            auto_specs = neural_specs_from_assignments(
                assignment_entries,
                claimed_ids,
                client=mlflow_client,
                cfg=master_cfg,
                config_dir=config_dir,
                settings=settings,
            )
            if auto_specs:
                cfg = replace(cfg, preconditioners=cfg.preconditioners + tuple(auto_specs))
            outcomes.extend(
                _run_comparison_from_config(
                    cfg, entry, topology, case_config_path, settings, force=params.force
                )
            )
    return outcomes
