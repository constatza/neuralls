"""Backend helpers for comparison workflows."""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
from dlkit.infrastructure.config.job_config import FitJobConfig
from loguru import logger
from mlflow.tracking import MlflowClient
from torchalg.utils.device import resolve_device

from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.composition.assignments.job_loader import load_experiment_job
from neuralls.composition.assignments.model_resolution import (
    AssignmentModelContext,
    TrainingLookup,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison._linear_system import prepare_resolved_linear_system
from neuralls.composition.comparison._preconditioner_setup import (
    SystemPreconditionerCache,
    matrix_key,
)
from neuralls.composition.comparison.comparison_run import (
    compare_preconditioners,
    execute_method_workload,
)
from neuralls.composition.comparison.config_assembler import resolve_comparison_config
from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.composition.comparison.input_gate import require_current_inputs
from neuralls.composition.comparison.models import (
    ComparisonOutcome,
    ComparisonParams,
    CompletedMethodWorkload,
    MatrixWorkload,
    MatrixWorkloadMember,
    PlannedSystem,
    ResolvedComparisonInput,
    ResolvedMethod,
)
from neuralls.composition.comparison.plan import plan_comparisons
from neuralls.composition.comparison.recipes import system_tag
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
from neuralls.composition.identity.training import training_identity
from neuralls.composition.tracking.run_specs import (
    ComparisonRunTags,
    build_child_comparison_tags,
    build_comparison_run_spec,
    build_session_run_spec,
)
from neuralls.domain.identity import StageIdentity
from neuralls.domain.solver.comparison_summary import (
    format_results_summary,
    summarize_best_combinations,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.domain.solver.ports import CostRecorder
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    ComparisonRegistryEntry,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    CheckpointRefBearing,
    NeuralCheckpointRef,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
    PreconditionerType,
    TrainedAssignmentRefConfig,
)
from neuralls.platform.config.models.preconditioner_family import preconditioner_family
from neuralls.platform.config.registry import get_assignment_binding, resolve_assignment_binding
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.artifacts import write_comparison_artifacts
from neuralls.platform.reporting.comparison_inputs import (
    ComparisonInputArtifacts,
    stage_comparison_inputs,
)
from neuralls.platform.tracking.artifact_access import (
    ArtifactLeaseManager,
    MlflowArtifactLeaseManager,
)
from neuralls.platform.tracking.comparison_tracking import (
    MLflowGroupReportSink,
    create_cost_recorder,
    log_comparison_artifact_uri,
    log_comparison_input_artifacts,
    log_comparison_run_params,
    log_linear_system_params,
    log_skipped_preconditioners,
    setup_comparison_tracking,
)
from neuralls.platform.tracking.extra_features import fetch_extra_feature_names
from neuralls.platform.tracking.mlflow import build_workflow_environment
from neuralls.platform.tracking.mlflow_client import log_comparison_artifacts_to_mlflow
from neuralls.platform.tracking.mlflow_store import MlflowIdentityStore
from neuralls.platform.tracking.model_registry import build_registered_model_name
from neuralls.shared.device import release_device_memory, warm_up_device
from neuralls.shared.digest import Digest, content_digest, digest_memo
from neuralls.shared.errors import StageInputsStale
from neuralls.shared.types import ComparisonRhsSourceKind, DatasetId, MatrixId


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


def _iter_checkpoint_refs(
    specs: Sequence[PreconditionerConfig],
) -> Iterator[tuple[PreconditionerConfig, str, NeuralCheckpointRef]]:
    """Yield every ``(spec, label, ref)`` triple reachable from these specs.

    The single traversal of the `CheckpointRefBearing` protocol used by every
    consumer that asks a question about a comparison's checkpoint dependencies
    — "does anything need resolving", "which assignments are referenced",
    "what identity do the resolved checkpoints have". Specs that bear no
    checkpoint ref (static preconditioners, an unfitted POD coarsening) simply
    contribute nothing.
    """
    for spec in specs:
        if not isinstance(spec, CheckpointRefBearing):
            continue
        for label, ref in spec.checkpoint_refs():
            yield spec, label, ref


def _needs_model_resolution(specs: tuple[PreconditionerConfig, ...]) -> bool:
    """Return True when any checkpoint-bearing preconditioner needs model_ref lookup.

    Generic over `CheckpointRefBearing` (not `NeuralPreconditionerConfig`-only)
    so a checkpoint-backed `AMGPreconditionerConfig` (POD-2G fitted ahead of
    time via a `FitJobConfig` assignment) triggers resolution the same way a
    neural preconditioner does — one dispatch point, no per-kind branching.
    """
    return any(ref.model_ref is not None for _, _, ref in _iter_checkpoint_refs(specs))


def _referenced_assignment_ids(
    specs: tuple[PreconditionerConfig, ...],
) -> tuple[str, ...]:
    """Return assignment ids explicitly referenced by checkpoint-bearing specs.

    Order is first-reference-wins, deduplicated — callers resolve one model
    context per id, and a stable order keeps their logs reproducible.
    """
    return tuple(
        dict.fromkeys(
            ref.assignment
            for _, _, ref in _iter_checkpoint_refs(specs)
            if ref.assignment is not None
        )
    )


def _assignment_training_identity(
    *,
    job_config_path: Path,
    data_config_path: Path,
    settings: NeurallsSettings,
) -> StageIdentity | None:
    """Derive an assignment's training identity, or ``None`` if its dataset is not ready.

    ``None`` makes strict model resolution fail for that assignment with a
    clear message (never a silent match against an unverifiable dataset).
    """
    try:
        return training_identity(
            job_config_path=job_config_path,
            data_config_path=data_config_path,
            settings=settings,
        )
    except FileNotFoundError, StageInputsStale:
        return None


def _build_master_assignment_contexts(
    assignment_ids: tuple[str, ...],
    settings: NeurallsSettings,
    *,
    master_cfg: CaseConfig,
    config_dir: Path,
) -> dict[str, AssignmentModelContext]:
    """Resolve dataset/model identity for all case-config assignments.

    Takes the case config and its resolution root directly, rather than a
    path to reload — callers that already have both in scope (the common
    case: this runs once per comparison entry needing model resolution)
    avoid re-reading and re-validating the same TOML file on every call.
    """
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
            training=TrainingLookup(
                experiment=master_cfg.names.training,
                identity=_assignment_training_identity(
                    job_config_path=binding.job_config_path,
                    data_config_path=binding.data_config_path,
                    settings=settings,
                ),
            ),
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
    return set(_referenced_assignment_ids(specs))


@dataclass(frozen=True)
class _AssignmentJobContext:
    """One assignment's resolved job settings plus its dataset directory."""

    job: AnyJobConfig
    dataset_dir: Path
    training: StageIdentity | None


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
    return _AssignmentJobContext(
        job=job,
        dataset_dir=settings.processed_dir / dataset_id,
        training=_assignment_training_identity(
            job_config_path=binding.job_config_path,
            data_config_path=binding.data_config_path,
            settings=settings,
        ),
    )


def _extra_input_names_of_training_run(
    context: _AssignmentJobContext, *, experiment: str, client: MlflowClient
) -> tuple[str, ...]:
    """Read extra input names from the very run strict resolution will select.

    Uses the same identity lookup as model resolution, so the names can never
    come from a different run than the checkpoint that gets loaded.
    """
    if context.training is None:
        return ()
    reused = MlflowIdentityStore(
        tracking_uri=client.tracking_uri, experiment=experiment, require_checkpoint=True
    ).find(context.training)
    if reused is None or reused.run_id is None:
        return ()
    return fetch_extra_feature_names(reused.run_id, client=client)


def _neural_spec_from_assignment(
    entry: AssignmentEntry,
    *,
    extra_input_names: tuple[str, ...],
) -> NeuralPreconditionerConfig:
    """Build an unresolved neural preconditioner stub for a train/search-kind assignment.

    The result key (`spec.name`) is derived from the config's own fields
    (`preconditioner.py::preconditioner_slug`), not set here — it always
    embeds `entry.id` via the `assignment` field below, so it stays
    traceable to the assignment without needing a separately-chosen name.
    References the assignment's training run through a strict
    `TrainedAssignmentRefConfig` (same experiment, FINISHED, checkpoint present,
    current dataset), rather than a registry entry or a tag search — the
    registry is reserved for deliberate/manual promotion. Also fetches the
    ``neuralls.extra_feature_names`` tag from the training run so that FiLM
    models receive their condition tensor during comparison.
    """
    return NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        assignment=entry.id,
        model_ref=TrainedAssignmentRefConfig(assignment_id=entry.id),
        extra_input_names=extra_input_names,
    )


def _pod_fit_spec_from_assignment(
    entry: AssignmentEntry,
    *,
    job: FitJobConfig,
    dataset_dir: Path,
) -> AMGPreconditionerConfig:
    """Build an unresolved AMG/POD stub for a `fit`-kind (POD-2G) assignment.

    Mirrors `_neural_spec_from_assignment`'s "derived result key, unresolved
    stub referencing the assignment's training run" shape, but for
    `AMGPreconditionerConfig(coarsening=PODCoarseningConfig(...))` instead of
    `NeuralPreconditionerConfig` — the checkpoint-shaped side of the kind
    dispatch in `neural_specs_from_assignments`. `dataset_dir`/`rank` are
    populated from the assignment's own job/dataset so the config stays
    valid (required fields) even before resolution runs; once
    `resolved_checkpoint_path` is set by
    `resolve_preconditioner_models_with_warnings`, the factory reconstructs
    the fitted basis from the checkpoint and never reads these.
    """
    rank = getattr(job.model, "rank", None)
    if rank is None:
        raise ValueError(
            f"Assignment '{entry.id}' job declares run.type='fit' but its model has no "
            "'rank' hyperparameter — expected a PODCoarseningStrategy-shaped model."
        )
    return AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(
            dataset_dir=dataset_dir,
            rank=rank,
            assignment=entry.id,
            model_ref=TrainedAssignmentRefConfig(assignment_id=entry.id),
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
        match context.job:
            case FitJobConfig():
                specs.append(
                    _pod_fit_spec_from_assignment(
                        entry, job=context.job, dataset_dir=context.dataset_dir
                    )
                )
            case _:
                specs.append(
                    _neural_spec_from_assignment(
                        entry,
                        extra_input_names=_extra_input_names_of_training_run(
                            context, experiment=cfg.names.training, client=client
                        ),
                    )
                )
    return specs


@dataclass(frozen=True)
class _ComparisonResolutionContext:
    """Everything needed to resolve one comparison entry's preconditioners.

    One instance is built once per `_run_comparison_from_config`/
    `_prepare_all_comparisons` call and threaded unchanged through
    `_prepare_comparison_entry` -> `_resolve_comparison_specs` ->
    `_resolve_specs`, instead of each of those functions separately
    re-declaring `topology`/`case_config_path`/`settings`/`master_cfg` as
    their own loose parameters.

    Attributes:
        master_cfg: The already-loaded case config, when the caller has one
            in scope (the batch path) — avoids reloading it from
            `case_config_path` for every comparison entry. Standalone
            callers with no preloaded config leave this unset and pay one
            lazy reload, only when an assignment-bound model ref actually
            needs it.
        artifact_leases: Lease manager shared by every entry of a sweep, owned
            (entered and closed) by the caller. It downloads each MLflow
            artifact once however many entries reference it, and keeps the
            files alive until the sweep ends. Standalone callers leave this
            unset and each entry then owns a private manager.
    """

    topology: ComparisonTopology
    case_config_path: Path
    settings: NeurallsSettings
    master_cfg: CaseConfig | None = None
    artifact_leases: MlflowArtifactLeaseManager | None = None


@dataclass(frozen=True)
class ResolvedComparisonSpecs:
    """One comparison's preconditioners after checkpoint resolution.

    The `checkpoint_digests` are always derived from `specs` at construction
    (see `from_resolution`), so the specs and the checkpoint content the
    comparison's identity covers can never drift apart.

    Attributes:
        specs: Preconditioner configs with concrete checkpoint paths baked in.
        warnings: Human-readable notes about specs that failed to resolve and
            were skipped.
        checkpoint_digests: Content digest of every resolved checkpoint the
            specs depend on, keyed by spec position and ref label.
    """

    specs: list[PreconditionerConfig]
    warnings: tuple[str, ...]
    checkpoint_digests: Mapping[str, Digest]

    @classmethod
    def from_resolution(
        cls, specs: list[PreconditionerConfig], warnings: tuple[str, ...]
    ) -> ResolvedComparisonSpecs:
        """Bundle a resolution result with the checkpoint digests it implies."""
        return cls(specs=specs, warnings=warnings, checkpoint_digests=_checkpoint_digests(specs))


def _resolve_specs(
    cfg: ComparisonConfig,
    context: _ComparisonResolutionContext,
    artifact_leases: ArtifactLeaseManager,
) -> ResolvedComparisonSpecs:
    """Resolve model_ref preconditioners into concrete checkpoint paths."""
    specs = _resolve_neural_preconditioners(list(cfg.preconditioners))
    if not _needs_model_resolution(cfg.preconditioners):
        return ResolvedComparisonSpecs.from_resolution(specs, ())

    assignment_ids = _referenced_assignment_ids(cfg.preconditioners)
    assignment_contexts = None
    if assignment_ids:
        resolved_master_cfg = (
            context.master_cfg
            if context.master_cfg is not None
            else load_validated_case_config(context.case_config_path, context.settings)[0]
        )
        assignment_contexts = _build_master_assignment_contexts(
            assignment_ids,
            context.settings,
            master_cfg=resolved_master_cfg,
            config_dir=context.case_config_path.resolve().parent,
        )
    resolution = resolve_preconditioner_models_with_warnings(
        specs=specs,
        tracking_uri=context.topology.model_store_tracking_uri,
        artifact_leases=artifact_leases,
        dataset_alias=cfg.general.data.dataset_alias,
        assignment_contexts=assignment_contexts,
        skip_unresolved=True,
    )
    return ResolvedComparisonSpecs.from_resolution(resolution.specs, resolution.warnings)


def _checkpoint_digests(specs: Sequence[PreconditionerConfig]) -> dict[str, Digest]:
    """Content digest of every resolved checkpoint the specs depend on.

    The comparison depends on the checkpoint *bytes* it actually loads, not on
    which run they came from, so a retrained model with a different result
    changes the identity and an identical one does not. Refs without a
    resolved checkpoint (an unfitted inline POD coarsening) contribute nothing;
    their fit data is covered by the spec's own identity fields.
    """
    digests: dict[str, Digest] = {}
    for index, spec in enumerate(specs):
        for _, label, ref in _iter_checkpoint_refs([spec]):
            checkpoint_path = ref.resolved_checkpoint_path
            if checkpoint_path is None:
                continue
            digest_key = f"{index}:{label}"
            logger.info(
                "Inspecting checkpoint '{}' for preconditioner '{}' before hashing.",
                checkpoint_path,
                spec.name,
            )
            checkpoint_size = checkpoint_path.stat().st_size
            logger.info(
                "Hashing checkpoint '{}' for preconditioner '{}' ({} bytes).",
                checkpoint_path,
                spec.name,
                checkpoint_size,
            )
            digests[digest_key] = content_digest(checkpoint_path)
            logger.info(
                "Finished hashing checkpoint for preconditioner '{}'.",
                spec.name,
            )
    return digests


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
    planned: PlannedSystem,
    work_root: Path,
    resolved_specs: list[PreconditionerConfig],
    settings: NeurallsSettings,
    cache: SystemPreconditionerCache | None = None,
) -> ComparisonResult:
    """Run comparison with already-resolved preconditioner checkpoint paths.

    `cache`, when given, lets a preconditioner built for an earlier sibling
    system in the same `comparison_sweeps` group be reused here instead of
    rebuilt — see `SystemPreconditionerCache`. `None` (the default) means
    this call builds every preconditioner fresh.
    """
    entry = planned.entry
    resolved_input = resolve_comparison_input(
        matrix_path=Path(cfg.general.data.matrix_path),
        matrix_dataset_id=entry.matrix_dataset,
        matrix_index=cfg.general.data.matrix_index,
        matrix_provenance=planned.provenance,
        require_non_residual_rhs=cfg.general.data.require_non_residual_rhs,
        seed=cfg.general.data.selection_seed,
        rhs_source_kind=_require_rhs_source_kind(cfg),
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
            row_kind=resolved_input.row_kind,
            rhs_source_kind=resolved_input.rhs_source_kind,
            rhs_source_params=resolved_input.rhs_source_params,
            matrix_provenance=resolved_input.matrix_provenance,
        ),
    )
    raw_result = compare_preconditioners(
        general_params=cfg.general,
        preconditioner_configs=resolved_specs,
        output_root=work_root,
        display_name=entry.effective_display_name,
        resolved_input=resolved_input,
        settings=settings,
        cache=cache,
        notation_policy=cfg.notation_policy,
    )
    write_comparison_artifacts(
        result=raw_result,
        work_root=work_root,
        comparison_config=entry.method,
    )
    log_comparison_input_artifacts(staged_input_dir)
    return raw_result


def _build_child_run_tags(
    result: ComparisonResult,
    *,
    comparison_id: str,
    parent_run_name: str,
) -> dict[str, dict[str, str]]:
    """Build the MLflow tag set for each per-preconditioner child run.

    Pure: derives tags from an already-computed result, touching no MLflow
    state, so the tag shape can be asserted without an active run.
    """
    return {
        name: build_child_comparison_tags(
            preconditioner_name=name,
            comparison_id=comparison_id,
            parent_run_name=parent_run_name,
        ).as_mlflow_tags()
        for name in result.results
    }


def _run_and_log_comparison(
    prepared: _PreparedComparisonExecution,
    *,
    run_name: str,
    comp_run_id: str,
    cost_recorder: CostRecorder | None = None,
    cache: SystemPreconditionerCache | None = None,
) -> ComparisonResult:
    """Run one comparison in a scratch directory, logging into the open MLflow run.

    Args:
        prepared: The entry, config, topology, and resolved specs to execute.
        run_name: Display name of the enclosing run, used as the child runs' parent label.
        comp_run_id: Id of the enclosing run that artifacts are uploaded to.
        cost_recorder: Where to record the comparison's cost/outcome metrics
            (DI for testing); defaults to MLflow, logging into the open run.
        cache: Optional per-sweep-group preconditioner cache (see
            `SystemPreconditionerCache`); `None` for a non-sweep entry.

    Returns:
        The comparison result, after its artifacts and metrics have been logged.
    """
    recorder = cost_recorder if cost_recorder is not None else create_cost_recorder()
    with tempfile.TemporaryDirectory() as tmp:
        work_root = Path(tmp)
        raw_result = _run_comparison_with_resolved_specs(
            cfg=prepared.cfg,
            planned=prepared.planned,
            work_root=work_root,
            resolved_specs=prepared.resolved.specs,
            settings=prepared.settings,
            cache=cache,
        )
        log_skipped_preconditioners(prepared.resolved.warnings)
        log_comparison_artifacts_to_mlflow(
            tracking_uri=prepared.topology.tracking_uri,
            run_id=comp_run_id,
            work_root=work_root,
        )
        recorder.record(
            raw_result,
            child_run_tags=_build_child_run_tags(
                raw_result, comparison_id=prepared.entry.id, parent_run_name=run_name
            ),
        )
        log_linear_system_params(raw_result)
    return raw_result


def _execute_comparison_in_run(
    prepared: _PreparedComparisonExecution,
    *,
    run_name: str,
    comp_tags: ComparisonRunTags,
    cache: SystemPreconditionerCache | None = None,
) -> ComparisonOutcome:
    """Open an MLflow run, execute the comparison, upload artifacts, and report the outcome.

    Args:
        prepared: Everything resolved ahead of execution — config, registry
            entry, topology, resolved specs, warnings, and dependency hash.
        run_name: Display name for the MLflow run.
        comp_tags: Structured tags applied to the MLflow run.
        cache: Optional per-sweep-group preconditioner cache (see
            `SystemPreconditionerCache`); `None` for a non-sweep entry.

    Returns:
        The successful ComparisonOutcome for this entry.
    """
    entry = prepared.entry
    with mlflow.start_run(
        run_name=run_name,
        nested=True,
        tags={**comp_tags.as_mlflow_tags(), **prepared.identity.tags()},
    ) as comp_run:
        comp_run_id = comp_run.info.run_id
        log_comparison_artifact_uri()
        raw_result = _run_and_log_comparison(
            prepared, run_name=run_name, comp_run_id=comp_run_id, cache=cache
        )
        log_comparison_run_params(
            comp_run_id=comp_run_id,
            comparison_id=entry.id,
            comparison_display_name=entry.effective_display_name,
            comparison_config=entry.method.stem if entry.method is not None else entry.id,
        )

    return _comparison_outcome(
        entry, success=True, payload=raw_result, warnings=prepared.resolved.warnings
    )


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
    context: _ComparisonResolutionContext,
    artifact_leases: ArtifactLeaseManager,
) -> ResolvedComparisonSpecs:
    """Resolve preconditioner checkpoints, rejecting a comparison with nothing left to run.

    Must run inside the same lease-manager scope as execution — leased
    checkpoint files are released when that scope closes.

    Raises:
        ValueError: When all preconditioners fail to resolve.
    """
    resolved = _resolve_specs(cfg, context, artifact_leases)
    if not resolved.specs:
        raise ValueError("No runnable preconditioners remain after model resolution.")
    return resolved


def _comparison_failure_outcome(
    entry: ComparisonRegistryEntry, exc: Exception
) -> ComparisonOutcome:
    """Shared 'resolve or execute raised' outcome — used by both prepare and execute."""
    logger.error("Comparison '{}' failed: {}", entry.id, exc)
    return _comparison_outcome(entry, success=False, error=str(exc))


@dataclass(frozen=True)
class _PreparedComparisonExecution:
    """One entry, resolved and confirmed to need real execution.

    `cleanup` owns the entry's artifact-lease lifetime (already resolved into
    `resolved.specs`' baked-in checkpoint paths) — the caller must enter it
    into its own ExitStack (or close it directly) once execution is done.
    """

    cfg: ComparisonConfig
    planned: PlannedSystem
    topology: ComparisonTopology
    cleanup: contextlib.ExitStack
    resolved: ResolvedComparisonSpecs
    identity: StageIdentity
    settings: NeurallsSettings

    @property
    def entry(self) -> ComparisonRegistryEntry:
        """The planned system's concrete registry entry."""
        return self.planned.entry


def _prepare_comparison_entry(
    cfg: ComparisonConfig,
    planned: PlannedSystem,
    context: _ComparisonResolutionContext,
    *,
    force: bool = False,
) -> ComparisonOutcome | _PreparedComparisonExecution:
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
    and a `_PreparedComparisonExecution` instead detaches that lifetime for the
    caller to keep open through execution.

    Args:
        cfg: Fully resolved ComparisonConfig with injected data paths and preconditioners.
        planned: The planned system, whose entry provides display name and method path.
        context: Shared resolution context (topology, case config, settings).
        force: Treat this entry as needing execution even if a matching comparison
            already completed successfully.

    Returns:
        A cache-hit/failure `ComparisonOutcome`, or a `_PreparedComparisonExecution`
        ready for `_execute_prepared_comparison`.
    """
    entry = planned.entry
    logger.info("Preparing comparison '{}': validating input sources.", entry.id)
    try:
        _validate_comparison_sources(cfg)
    except (FileNotFoundError, ValueError) as exc:
        return _comparison_failure_outcome(entry, exc)
    logger.info("Comparison '{}': input-source validation complete.", entry.id)

    with contextlib.ExitStack() as local_stack:
        artifact_leases = context.artifact_leases or local_stack.enter_context(
            MlflowArtifactLeaseManager(
                client=MlflowClient(tracking_uri=context.topology.model_store_tracking_uri)
            )
        )
        logger.info("Comparison '{}': resolving preconditioner checkpoints.", entry.id)
        try:
            resolved = _resolve_comparison_specs(cfg, context, artifact_leases)
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            return _comparison_failure_outcome(entry, exc)
        logger.info(
            "Comparison '{}': resolved {} preconditioners with {} warnings.",
            entry.id,
            len(resolved.specs),
            len(resolved.warnings),
        )

        logger.info(
            "Comparison '{}': building comparison identity, including matrix/dataset content.",
            entry.id,
        )
        identity = comparison_identity(cfg, resolved.specs, resolved.checkpoint_digests)
        logger.info("Comparison '{}': built identity {}.", entry.id, identity.key)
        logger.info("Comparison '{}': checking MLflow for a reusable run.", entry.id)
        reused = gate_reuse(
            MlflowIdentityStore(
                tracking_uri=context.topology.tracking_uri,
                experiment=context.topology.experiment_name,
            ),
            identity,
            force=force,
            label=entry.id,
        )
        if reused is not None:
            logger.info(
                "Comparison '{}': reuse lookup selected run {}.",
                entry.id,
                reused.run_id,
            )
            return _comparison_outcome(entry, success=True)
        logger.info("Comparison '{}': no reusable run selected; execution is required.", entry.id)

        # Needs real execution: detach the lease manager's cleanup from this
        # function's scope-exit and hand it to the caller instead.
        return _PreparedComparisonExecution(
            cfg=cfg,
            planned=planned,
            topology=context.topology,
            cleanup=local_stack.pop_all(),
            resolved=resolved,
            identity=identity,
            settings=context.settings,
        )


def _execute_prepared_comparison(
    prepared: _PreparedComparisonExecution,
    *,
    cache: SystemPreconditionerCache | None = None,
) -> ComparisonOutcome:
    """Build the run spec/tags and execute inside an already-active nested MLflow run.

    Args:
        prepared: Everything resolved ahead of execution.
        cache: Optional per-sweep-group preconditioner cache (see
            `SystemPreconditionerCache`); `None` for a non-sweep entry.
    """
    try:
        run_name, comp_tags = build_comparison_run_spec(
            entry=prepared.entry, include_timestamp=False
        )
        return _execute_comparison_in_run(
            prepared, run_name=run_name, comp_tags=comp_tags, cache=cache
        )
    except (ValueError, RuntimeError, KeyError, IndexError) as exc:
        return _comparison_failure_outcome(prepared.entry, exc)


def _run_comparison_from_config(
    cfg: ComparisonConfig,
    planned: PlannedSystem,
    topology: ComparisonTopology,
    case_config_path: Path,
    settings: NeurallsSettings,
    *,
    force: bool = False,
) -> list[ComparisonOutcome]:
    """Resolve, reuse-check, and execute one comparison, logging results to MLflow.

    Standalone single-entry entry point: runs nested under an already-active
    MLflow run (the caller's session parent, if any), so this becomes the
    "subrun" level between the case-wide parent and per-preconditioner leaf
    runs. `run_comparison_batch` does not call this directly — it composes
    `_prepare_comparison_entry`/`_execute_prepared_comparison` itself so it
    can gate the whole batch's session parent run on whether any entry needs
    real execution.

    Args:
        cfg: Fully resolved ComparisonConfig with injected data paths and preconditioners.
        planned: The planned system, whose entry provides display name and method path.
        topology: Resolved MLflow topology for the whole batch (tracking/experiment).
        case_config_path: Case config path for model resolution context.
        settings: Resolved runtime settings.
        force: Rerun even if a matching comparison already completed successfully.

    Returns:
        Single-element list with the outcome of the comparison run.
    """
    context = _ComparisonResolutionContext(
        topology=topology, case_config_path=case_config_path, settings=settings
    )
    prepared = _prepare_comparison_entry(cfg, planned, context, force=force)
    if isinstance(prepared, ComparisonOutcome):
        return [prepared]
    with prepared.cleanup:
        return [_execute_prepared_comparison(prepared)]


def _validate_comparison_sources(cfg: ComparisonConfig) -> None:
    """Validate only the concrete input artifacts required by this workflow mode.

    Dispatches through the same `ComparisonSourceHandler` registry that
    resolves the inputs, so preflight validation and resolution can never
    disagree about which kinds are supported or which files a kind reads.
    """
    validate_comparison_source(
        ComparisonSourceSpec(
            matrix_path=Path(cfg.general.data.matrix_path),
            rhs_source_kind=_require_rhs_source_kind(cfg),
            rhs_source_params=cfg.general.data.rhs_source_params or {},
        )
    )


def _prepare_all_comparisons(
    master_cfg: CaseConfig,
    topology: ComparisonTopology,
    case_config_path: Path,
    settings: NeurallsSettings,
    mlflow_client: MlflowClient,
    stack: contextlib.ExitStack,
    *,
    force: bool,
) -> tuple[list[ComparisonOutcome | _PreparedComparisonExecution], tuple[PlannedSystem, ...]]:
    """Plan, resolve and reuse-check every system of the case.

    Opens no MLflow run. `plan_comparisons` turns every `[[comparisons]]` and
    `[[comparison_sweeps]]` entry into concrete systems first, so each is
    resolved and reuse-checked identically. Systems needing execution have
    their cleanup registered onto `stack`, which the caller must keep open
    until after execution.

    Returns:
        One outcome-or-prepared-execution per planned system, in plan order,
        plus the planned systems themselves.
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
    context = _ComparisonResolutionContext(
        topology=topology,
        case_config_path=case_config_path,
        settings=settings,
        master_cfg=master_cfg,
        artifact_leases=stack.enter_context(
            MlflowArtifactLeaseManager(
                client=MlflowClient(tracking_uri=topology.model_store_tracking_uri)
            )
        ),
    )
    pending: list[ComparisonOutcome | _PreparedComparisonExecution] = []
    planned_systems = plan_comparisons(master_cfg, config_dir, settings)
    for planned in planned_systems:
        entry = planned.entry
        logger.info(f"Comparison [{entry.id}]: {entry.effective_display_name}")
        assignment_entries: list[AssignmentEntry] = (
            [e for e in master_cfg.assignments if e.id in entry.assignments]
            if entry.assignments
            else list(master_cfg.assignments)
        )
        cfg = resolve_comparison_config(master_cfg, config_dir, planned, settings)
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
        validate_unique_preconditioner_keys(cfg.preconditioners)

        prepared = _prepare_comparison_entry(cfg, planned, context, force=force)
        if isinstance(prepared, _PreparedComparisonExecution):
            stack.enter_context(prepared.cleanup)
        pending.append(prepared)
    validate_dataset_specs_consistent(planned_systems, _identities_by_entry_id(pending))
    return pending, planned_systems


def _identities_by_entry_id(
    pending: Sequence[ComparisonOutcome | _PreparedComparisonExecution],
) -> dict[str, dict[str, Digest]]:
    """Identity of every preconditioner of every entry that will execute, by name."""
    return {
        p.entry.id: preconditioner_identities(p.resolved.specs, p.resolved.checkpoint_digests)
        for p in pending
        if isinstance(p, _PreparedComparisonExecution)
    }


def _dataset_of(prepared: _PreparedComparisonExecution) -> DatasetId:
    return DatasetId(prepared.entry.matrix_dataset)


def _matrix_of(prepared: _PreparedComparisonExecution) -> tuple[DatasetId, MatrixId]:
    return _dataset_of(prepared), matrix_key(prepared.planned)


def _method_major_schedule(
    pending: Sequence[ComparisonOutcome | _PreparedComparisonExecution],
) -> tuple[
    tuple[
        ResolvedMethod,
        tuple[
            tuple[
                tuple[DatasetId, MatrixId],
                tuple[_PreparedComparisonExecution, ...],
            ],
            ...,
        ],
    ],
    ...,
]:
    """Group executable systems as method -> matrix -> ordered RHS members."""
    methods: dict[tuple[str, Digest], ResolvedMethod] = {}
    grouped: dict[
        tuple[str, Digest],
        dict[tuple[DatasetId, MatrixId], list[_PreparedComparisonExecution]],
    ] = {}
    for item in pending:
        if not isinstance(item, _PreparedComparisonExecution):
            continue
        identities = preconditioner_identities(
            item.resolved.specs, item.resolved.checkpoint_digests
        )
        for config in item.resolved.specs:
            method_key = config.name, identities[config.name]
            methods.setdefault(method_key, ResolvedMethod(config))
            grouped.setdefault(method_key, {}).setdefault(_matrix_of(item), []).append(item)
    return tuple(
        (
            method,
            tuple((matrix, tuple(systems)) for matrix, systems in grouped[key].items()),
        )
        for key, method in methods.items()
    )


def _load_matrix_workloads(
    matrices: Sequence[
        tuple[
            tuple[DatasetId, MatrixId],
            tuple[_PreparedComparisonExecution, ...],
        ]
    ],
) -> tuple[MatrixWorkload, ...]:
    """Read one method's matrices/RHS values before online timing begins."""
    bundles: list[MatrixWorkload] = []
    for (dataset, matrix_id), systems in matrices:
        loaded: list[tuple[_PreparedComparisonExecution, ResolvedComparisonInput]] = []
        for prepared in systems:
            cfg = prepared.cfg
            resolved = resolve_comparison_input(
                matrix_path=Path(cfg.general.data.matrix_path),
                matrix_dataset_id=prepared.entry.matrix_dataset,
                matrix_index=cfg.general.data.matrix_index,
                matrix_provenance=prepared.planned.provenance,
                require_non_residual_rhs=cfg.general.data.require_non_residual_rhs,
                seed=cfg.general.data.selection_seed,
                rhs_source_kind=_require_rhs_source_kind(cfg),
                rhs_source_params=cfg.general.data.rhs_source_params,
            )
            system = prepare_resolved_linear_system(resolved, cfg.general.data.normalize_system)
            loaded.append(
                (
                    prepared,
                    replace(
                        resolved,
                        matrix=system.matrix.numpy(),
                        rhs=system.rhs.numpy(),
                    ),
                )
            )
        first_prepared, first_input = loaded[0]
        bundles.append(
            MatrixWorkload(
                dataset=dataset,
                matrix_id=matrix_id,
                matrix=first_input.matrix,
                matrix_path=Path(first_prepared.cfg.general.data.matrix_path),
                matrix_index=first_prepared.cfg.general.data.matrix_index,
                members=tuple(
                    MatrixWorkloadMember(
                        tag=system_tag(prepared.planned),
                        rhs=resolved.rhs,
                    )
                    for prepared, resolved in loaded
                ),
            )
        )
    return tuple(bundles)


def _outcomes_from_workloads(
    pending: Sequence[ComparisonOutcome | _PreparedComparisonExecution],
    workloads: Sequence[CompletedMethodWorkload],
) -> list[ComparisonOutcome]:
    """Project canonical diagnostics back into the retained per-system API."""
    diagnostics: dict[str, dict[str, CGComparisonResult]] = {}
    for workload in workloads:
        for system_id, result in workload.diagnostics:
            diagnostics.setdefault(str(system_id), {})[str(workload.method)] = result

    outcomes: list[ComparisonOutcome] = []
    for item in pending:
        if isinstance(item, ComparisonOutcome):
            outcomes.append(item)
            continue
        results = diagnostics.get(item.entry.id, {})
        if not results:
            outcomes.append(
                _comparison_failure_outcome(
                    item.entry, RuntimeError("method workload produced no diagnostics")
                )
            )
            continue
        typed_results = dict(results)
        styles = {
            config.name: PreconditionerPlotStyle(
                label=config.name,
                family=preconditioner_family(config),
            )
            for config in item.resolved.specs
            if config.name in typed_results
        }
        outcomes.append(
            _comparison_outcome(
                item.entry,
                success=True,
                warnings=item.resolved.warnings,
                payload=ComparisonResult(
                    results=typed_results,
                    summary=format_results_summary(typed_results),
                    solver_params=item.cfg.general,
                    preconditioners=tuple(typed_results),
                    recommendations=summarize_best_combinations(typed_results),
                    rhs_source_kind=item.cfg.general.data.rhs_source_kind,
                    plot_styles=styles,
                ),
            )
        )
    return outcomes


def _execute_all(
    pending: Sequence[ComparisonOutcome | _PreparedComparisonExecution],
) -> tuple[list[ComparisonOutcome], tuple[CompletedMethodWorkload, ...]]:
    """Execute pending systems in method -> matrix -> ordered RHS order."""
    warm_up_device(resolve_device())
    workloads: list[CompletedMethodWorkload] = []
    for method, matrices in _method_major_schedule(pending):
        bundles = _load_matrix_workloads(matrices)
        first = matrices[0][1][0]
        workload = execute_method_workload(
            method,
            bundles,
            params=first.cfg.general.params,
            settings=first.settings,
            notation_policy=first.cfg.notation_policy,
        )
        workloads.append(workload)
        release_device_memory()
    return _outcomes_from_workloads(pending, workloads), tuple(workloads)


def run_comparison_batch(
    case_config_path: Path,
    params: ComparisonParams,
    settings: NeurallsSettings | None = None,
) -> list[ComparisonOutcome]:
    """Run all configured comparison entries from the case config.

    Every entry is resolved and reuse-checked first, with no MLflow run
    opened yet. Only if at least one entry genuinely needs execution does
    this open one MLflow session parent run for the whole batch — named
    ``"<case-name> | <timestamp>"``, mirroring training's session parent — so
    every executing entry's own run nests underneath it as a "subrun", with
    each entry's per-preconditioner runs nested one level further. If every
    entry is a cache hit, no MLflow run is created at all.

    Args:
        case_config_path: Path to the case config TOML.
        params: Comparison execution parameters (`force`).
        settings: Optional pre-loaded runtime settings.

    Returns:
        List of comparison outcomes, one per planned system, in plan order
        (sweep systems follow all [[comparisons]] systems).
    """
    settings = require_settings(settings, case_config_path=case_config_path)
    master_cfg, _ = _load_master_config(case_config_path, settings)
    if not master_cfg.comparisons and not master_cfg.comparison_sweeps:
        raise ValueError(
            "Case config must define at least one [[comparisons]] or [[comparison_sweeps]] entry."
        )

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

    with contextlib.ExitStack() as stack:
        stack.enter_context(digest_memo())
        pending, _planned_systems = _prepare_all_comparisons(
            master_cfg,
            topology,
            case_config_path,
            settings,
            mlflow_client,
            stack,
            force=params.force,
        )

        if not any(isinstance(p, _PreparedComparisonExecution) for p in pending):
            outcomes = [p for p in pending if isinstance(p, ComparisonOutcome)]
            report_all_levels((), MLflowGroupReportSink())
            return outcomes

        with mlflow.start_run(run_name=session_run_name, tags=session_tags.as_mlflow_tags()):
            outcomes, workloads = _execute_all(pending)
            report_all_levels(workloads, MLflowGroupReportSink())
            return outcomes
