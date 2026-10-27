"""Backend helpers for comparison workflows."""

from __future__ import annotations

import contextlib
import tempfile
import tomllib
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, assert_never

import mlflow
from dlkit.infrastructure.config.job_config import FitJobConfig
from loguru import logger
from mlflow.tracking import MlflowClient
from torchalg.utils.device import resolve_device

from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.composition.assignments.assignment_subjects import (
    neural_spec_from_assignment,
    pod_fit_spec_from_assignment,
    resolve_assignment_inputs,
)
from neuralls.composition.assignments.model_resolution import (
    AssignmentModelContext,
    TrainingLookup,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.composition.assignments.presentation_preflight import (
    compose_comparison_title,
    prepare_comparison_presentation,
)
from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison._linear_system import prepare_resolved_linear_system
from neuralls.composition.comparison._preconditioner_setup import (
    SystemPreconditionerCache,
    matrix_key,
)
from neuralls.composition.comparison.comparison_run import (
    build_comparison_result_from_diagnostics,
    compare_preconditioners,
    execute_method_workload,
)
from neuralls.composition.comparison.config_assembler import (
    resolve_comparison_config,
    resolve_dataset_dir,
)
from neuralls.composition.comparison.group_reporting import report_all_levels
from neuralls.composition.comparison.input_gate import require_current_inputs
from neuralls.composition.comparison.models import (
    ComparisonCompleted,
    ComparisonEntryFailed,
    ComparisonOutcome,
    ComparisonParams,
    ComparisonPresentation,
    ComparisonReused,
    CompletedMethodWorkload,
    MatrixWorkload,
    MatrixWorkloadMember,
    PlannedSystem,
    ResolvedMethod,
    UndescribedComparison,
    canonical_method_config,
    is_comparison_outcome,
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
    build_benchmark_method_run_spec,
    build_child_comparison_tags,
    build_comparison_run_spec,
    build_session_run_spec,
    comparison_run,
)
from neuralls.domain.benchmarking.selection import (
    NoReferenceConfigured,
    ReferenceSelection,
    resolve_reference_selection,
    validate_reference_preconditioner,
)
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import PreparedComparisonPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import NamedRun, TrackedRun, Untimed
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
    session_title,
)
from neuralls.domain.solver.models.benchmark_events import MeasuredPeakMemory, SetupCompleted
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.domain.solver.ports import BenchmarkMethodSink, CostRecorder
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    ComparisonRegistryEntry,
)
from neuralls.platform.config.models.preconditioner import (
    CheckpointRefBearing,
    NeuralCheckpointRef,
    NeuralPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.platform.config.registry import get_assignment_binding
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.artifacts import write_comparison_artifacts
from neuralls.platform.reporting.comparison_inputs import (
    ComparisonInputArtifacts,
    stage_comparison_inputs,
)
from neuralls.platform.reporting.figure_metadata import (
    InvalidFigureManifestError,
    validate_figure_manifest,
)
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.platform.tracking.artifact_access import (
    ArtifactLeaseManager,
    MlflowArtifactLeaseManager,
)
from neuralls.platform.tracking.comparison_tracking import (
    GROUP_REPORT_ARTIFACT_DIR,
    MLflowBenchmarkMethodSink,
    MLflowGroupReportSink,
    create_cost_recorder,
    log_comparison_artifact_uri,
    log_comparison_input_artifacts,
    log_comparison_run_params,
    log_configured_method_facts,
    log_linear_system_params,
    log_skipped_preconditioners,
    setup_comparison_tracking,
)
from neuralls.platform.tracking.extra_features import fetch_extra_feature_names
from neuralls.platform.tracking.failure_reporting import MLflowWorkflowFailureSink
from neuralls.platform.tracking.mlflow import build_workflow_environment
from neuralls.platform.tracking.mlflow_client import log_comparison_artifacts_to_mlflow
from neuralls.platform.tracking.mlflow_store import (
    MlflowIdentityStore,
    find_reusable_benchmark_method,
)
from neuralls.platform.tracking.model_registry import build_registered_model_name
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.shared.device import release_device_memory, warm_up_device
from neuralls.shared.digest import Digest, canonical_digest, content_digest, digest_memo
from neuralls.shared.errors import StageInputsStale
from neuralls.shared.scientific_description.models import ValidatedLabel
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import (
    ComparisonRhsSourceKind,
    DatasetId,
    MatrixId,
    PreconditionerName,
    SystemId,
)
from neuralls.shared.workflow_failures import (
    ComparisonBatchFailureContext,
    ComparisonBatchPhase,
    ComparisonMatrixFailureContext,
    ComparisonMatrixPhase,
    ComparisonMethodFailureContext,
    ComparisonMethodPhase,
    WorkflowFailureContext,
    WorkflowFailureSink,
)


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class _AssignmentJobContext:
    """One assignment's resolved job settings plus its dataset directory."""

    job: AnyJobConfig
    dataset_dir: Path
    training: StageIdentity | None
    architecture: ValidatedLabel | None
    """The job registry entry's explicit ``architecture``; None when it declares none."""


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
    inputs = resolve_assignment_inputs(entry, cfg=cfg, config_dir=config_dir, settings=settings)
    return _AssignmentJobContext(
        job=inputs.job,
        architecture=inputs.binding.architecture,
        dataset_dir=inputs.dataset_dir,
        training=_assignment_training_identity(
            job_config_path=inputs.binding.job_config_path,
            data_config_path=inputs.binding.data_config_path,
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
                    pod_fit_spec_from_assignment(
                        entry, job=context.job, dataset_dir=context.dataset_dir
                    )
                )
            case _:
                specs.append(
                    neural_spec_from_assignment(
                        entry,
                        extra_input_names=_extra_input_names_of_training_run(
                            context, experiment=cfg.names.training, client=client
                        ),
                        architecture=context.architecture,
                    )
                )
    return specs


@dataclass(frozen=True, slots=True)
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
    reuse_comparisons: bool = True


@dataclass(frozen=True, slots=True)
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
    presentation: ComparisonPresentation,
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
        resolved_input=resolved_input,
        settings=settings,
        cache=cache,
        presentation=presentation,
    )
    write_comparison_artifacts(
        result=raw_result,
        work_root=work_root,
        comparison_config=entry.method,
    )
    log_comparison_input_artifacts(staged_input_dir)
    return raw_result


def _build_child_runs(
    result: ComparisonResult,
    *,
    prepared: PreparedComparisonExecution,
    parent: NamedRun,
) -> dict[str, TrackedRun]:
    """Build the nested run (typed name and tags) of each per-preconditioner child run.

    Pure: derives the runs from an already-computed result and the prepared cohort,
    touching no MLflow state, so the naming can be asserted without an active run. Each
    child is named after its method's own presentation, never after its result key.
    """
    notation = prepared.cfg.notation_policy
    children: dict[str, TrackedRun] = {}
    for name in result.results:
        tags = build_child_comparison_tags(
            prepared=prepared.presentation.method(PreconditionerName(name)),
            notation=notation,
            comparison_id=prepared.entry.id,
            parent=parent,
        )
        children[name] = TrackedRun(tags.method, tags.as_mlflow_tags())
    return children


def _run_and_log_comparison(
    prepared: PreparedComparisonExecution,
    *,
    run: NamedRun,
    comp_run_id: str,
    cost_recorder: CostRecorder | None = None,
    cache: SystemPreconditionerCache | None = None,
) -> ComparisonResult:
    """Run one comparison in a scratch directory, logging into the open MLflow run.

    Args:
        prepared: The entry, config, topology, and resolved specs to execute.
        run: Typed name of the enclosing run, linked from each child run's tags.
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
            presentation=prepared.comparison_presentation,
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
            child_runs=_build_child_runs(raw_result, prepared=prepared, parent=run),
        )
        log_linear_system_params(raw_result)
    return raw_result


def _execute_comparison_in_run(
    prepared: PreparedComparisonExecution,
    *,
    run: NamedRun,
    comp_tags: ComparisonRunTags,
    cache: SystemPreconditionerCache | None = None,
) -> ComparisonOutcome:
    """Open an MLflow run, execute the comparison, upload artifacts, and report the outcome.

    Args:
        prepared: Everything resolved ahead of execution — config, registry
            entry, topology, resolved specs, warnings, and dependency hash.
        run: Typed name of the MLflow run.
        comp_tags: Structured tags applied to the MLflow run.
        cache: Optional per-sweep-group preconditioner cache (see
            `SystemPreconditionerCache`); `None` for a non-sweep entry.

    Returns:
        The successful ComparisonOutcome for this entry.
    """
    entry = prepared.entry
    with mlflow.start_run(
        run_name=run_name_text(run),
        nested=True,
        tags={**comp_tags.as_mlflow_tags(), **prepared.identity.tags()},
    ) as comp_run:
        comp_run_id = comp_run.info.run_id
        log_comparison_artifact_uri()
        raw_result = _run_and_log_comparison(
            prepared, run=run, comp_run_id=comp_run_id, cache=cache
        )
        log_comparison_run_params(
            comp_run_id=comp_run_id,
            comparison_id=entry.id,
            title=prepared.title,
            notation=prepared.cfg.notation_policy,
            comparison_config=entry.method.stem if entry.method is not None else entry.id,
        )

    return ComparisonCompleted(
        comparison_id=entry.id,
        run=run,
        payload=raw_result,
        warnings=prepared.resolved.warnings,
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


def _undescribed_failure(entry: ComparisonRegistryEntry, exc: Exception) -> ComparisonOutcome:
    """The outcome of a comparison that failed before its methods and dataset were resolved."""
    logger.error("A comparison failed before it could be described: {}", exc)
    return ComparisonEntryFailed(
        comparison_id=entry.id, run=UndescribedComparison(), reason=str(exc)
    )


def _described_failure(prepared: PreparedComparisonExecution, exc: Exception) -> ComparisonOutcome:
    """The outcome of a described comparison whose execution raised."""
    run = prepared.run_name
    logger.error("{} failed: {}", run_name_text(run), exc)
    return ComparisonEntryFailed(comparison_id=prepared.entry.id, run=run, reason=str(exc))


@dataclass(frozen=True, slots=True)
class PreparedComparisonExecution:
    """One entry, resolved and confirmed to need real execution.

    `cleanup` owns the entry's artifact-lease lifetime (already resolved into
    `resolved.specs`' baked-in checkpoint paths) — the caller must enter it
    into its own ExitStack (or close it directly) once execution is done.

    The presentation must describe exactly the resolved methods, in order, and a title
    composed from the dataset scope must be that presentation's own scope.
    """

    cfg: ComparisonConfig
    planned: PlannedSystem
    topology: ComparisonTopology
    cleanup: contextlib.ExitStack
    resolved: ResolvedComparisonSpecs
    identity: StageIdentity
    settings: NeurallsSettings
    presentation: PreparedComparisonPresentation
    """The validated scientific presentation of every method; execution requires it."""
    title: ComparisonTitle
    reference_selection: ReferenceSelection = field(default_factory=NoReferenceConfigured)

    def __post_init__(self) -> None:
        """Require the presentation and title to describe this execution's own methods."""
        spec_names = tuple(spec.name for spec in self.resolved.specs)
        method_ids = tuple(method.method_id for method in self.presentation.methods)
        if method_ids != spec_names:
            raise ValueError(
                f"presentation methods {method_ids!r} differ from the resolved specs "
                f"{spec_names!r}; they must be the same methods in the same order"
            )
        match self.title:
            case ComparisonScopePresentation() as scope:
                if scope != self.presentation.scope:
                    raise ValueError("a scope title must be the presentation's own scope")
            case EditorialPresentation():
                pass
            case _ as unreachable:
                assert_never(unreachable)

    @property
    def entry(self) -> ComparisonRegistryEntry:
        """The planned system's concrete registry entry."""
        return self.planned.entry

    @property
    def run_name(self) -> NamedRun:
        """The typed name of this system's comparison run, as MLflow names it."""
        return comparison_run(
            self.planned,
            self.title,
            self.presentation.scope.dataset,
            self.cfg.notation_policy,
            Untimed(),
        )

    @property
    def comparison_presentation(self) -> ComparisonPresentation:
        """The validated cohort, its title, and the case's notation, for the figures."""
        return ComparisonPresentation(
            methods=self.presentation, title=self.title, notation=self.cfg.notation_policy
        )


type PendingComparison = ComparisonOutcome | PreparedComparisonExecution
"""A planned system's state before execution: already decided, or ready to execute."""


def _planned_system_matches(item: PendingComparison, planned: PlannedSystem) -> bool:
    """Whether ``item`` is the state of ``planned`` and not of some other system."""
    match item:
        case PreparedComparisonExecution():
            return item.planned is planned
        case ComparisonCompleted() | ComparisonReused() | ComparisonEntryFailed():
            return item.comparison_id == planned.entry.id
        case _ as unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class PreparedComparisonBatch:
    """Every comparison of a case, resolved and presentation-validated; nothing has run.

    Session tracking and execution accept only this type, so an unvalidated cohort cannot
    reach MLflow or a numerical kernel. A comparison that is a reuse hit or a failure
    carries its outcome; one that must execute carries its prepared execution. Item ``i``
    is the state of planned system ``i``.
    """

    items: tuple[PendingComparison, ...]
    planned: tuple[PlannedSystem, ...]

    def __post_init__(self) -> None:
        """Require item ``i`` to be the state of planned system ``i``."""
        if len(self.items) != len(self.planned):
            raise ValueError("A prepared batch needs exactly one item per planned system")
        for position, (item, planned) in enumerate(zip(self.items, self.planned, strict=True)):
            if not _planned_system_matches(item, planned):
                raise ValueError(
                    f"A prepared batch item {position} is not the state of its planned system "
                    f"{planned.entry.id!r}"
                )


def _matrix_scope(
    planned: PlannedSystem, context: _ComparisonResolutionContext
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
    return ComparisonScopePresentation(load_dataset_presentation(dataset_dir))


def _prepare_comparison_entry(
    cfg: ComparisonConfig,
    planned: PlannedSystem,
    context: _ComparisonResolutionContext,
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
        ready for `_execute_prepared_comparison`.
    """
    entry = planned.entry
    logger.info("Preparing comparison '{}': validating input sources.", entry.id)
    try:
        _validate_comparison_sources(cfg)
    except (FileNotFoundError, ValueError) as exc:
        return _undescribed_failure(entry, exc)
    logger.info("Comparison '{}': input-source validation complete.", entry.id)
    configured_reference = cfg.general.reference_preconditioner
    if not isinstance(configured_reference, str):
        configured_reference = None
    validate_reference_preconditioner(
        configured_reference,
        tuple(spec.name for spec in cfg.preconditioners),
    )

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
            return _undescribed_failure(entry, exc)
        logger.info(
            "Comparison '{}': resolved {} preconditioners with {} warnings.",
            entry.id,
            len(resolved.specs),
            len(resolved.warnings),
        )
        skipped_reference = (
            {
                str(configured_reference): next(
                    warning for warning in resolved.warnings if configured_reference in warning
                )
            }
            if configured_reference is not None
            and any(configured_reference in warning for warning in resolved.warnings)
            else {}
        )
        reference_selection = resolve_reference_selection(
            configured_reference,
            tuple(spec.name for spec in resolved.specs),
            skipped=skipped_reference,
        )

        scope = _matrix_scope(planned, context)
        presentation = prepare_comparison_presentation(resolved.specs, scope)
        title = compose_comparison_title(entry.display_name, scope.dataset)
        run = comparison_run(planned, title, scope.dataset, cfg.notation_policy, Untimed())
        label = run_name_text(run)

        logger.info("{}: building comparison identity, including matrix/dataset content.", label)
        identity = comparison_identity(cfg, resolved.specs, resolved.checkpoint_digests)
        logger.info("{}: built identity {}.", label, identity.key)
        logger.info("{}: checking MLflow for a reusable run.", label)
        if context.reuse_comparisons:
            reused = gate_reuse(
                MlflowIdentityStore(
                    tracking_uri=context.topology.tracking_uri,
                    experiment=context.topology.experiment_name,
                ),
                identity,
                force=force,
                label=label,
            )
            if reused is not None:
                logger.info(
                    "{}: reuse lookup selected run {}.",
                    label,
                    reused.run_id,
                )
                return ComparisonReused(
                    comparison_id=entry.id,
                    run=run,
                )
            logger.info("{}: no reusable run selected; execution is required.", label)
        else:
            logger.info("{}: per-system reuse is disabled for batch execution.", label)

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


def _execute_prepared_comparison(
    prepared: PreparedComparisonExecution,
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
        run, comp_tags = build_comparison_run_spec(
            entry=prepared.entry,
            planned=prepared.planned,
            title=prepared.title,
            dataset=prepared.presentation.scope.dataset,
            notation=prepared.cfg.notation_policy,
            include_timestamp=False,
        )
        return _execute_comparison_in_run(prepared, run=run, comp_tags=comp_tags, cache=cache)
    except (ValueError, RuntimeError, KeyError, IndexError) as exc:
        return _described_failure(prepared, exc)


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
    if is_comparison_outcome(prepared):
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
        reuse_comparisons=reuse_comparisons,
    )
    pending: list[PendingComparison] = []
    planned_systems = plan_comparisons(master_cfg, config_dir, settings)
    for planned in planned_systems:
        entry = planned.entry
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
        if isinstance(prepared, PreparedComparisonExecution):
            stack.enter_context(prepared.cleanup)
        pending.append(prepared)
    validate_dataset_specs_consistent(planned_systems, _identities_by_entry_id(pending))
    return PreparedComparisonBatch(items=tuple(pending), planned=planned_systems)


def _identities_by_entry_id(
    pending: Sequence[PendingComparison],
) -> dict[str, dict[str, Digest]]:
    """Identity of every preconditioner of every entry that will execute, by name."""
    return {
        p.entry.id: preconditioner_identities(p.resolved.specs, p.resolved.checkpoint_digests)
        for p in pending
        if isinstance(p, PreparedComparisonExecution)
    }


def _dataset_of(prepared: PreparedComparisonExecution) -> DatasetId:
    return DatasetId(prepared.entry.matrix_dataset)


def _matrix_of(prepared: PreparedComparisonExecution) -> tuple[DatasetId, MatrixId]:
    return _dataset_of(prepared), matrix_key(prepared.planned)


def _method_major_schedule(
    pending: Sequence[PendingComparison],
) -> tuple[
    tuple[
        ResolvedMethod,
        tuple[
            tuple[
                tuple[DatasetId, MatrixId],
                tuple[PreparedComparisonExecution, ...],
            ],
            ...,
        ],
    ],
    ...,
]:
    """Group executable systems as method -> matrix -> ordered RHS members."""
    methods: dict[tuple[str, Digest], ResolvedMethod] = {}
    identities_by_name: dict[str, Digest] = {}
    semantics_by_name: dict[str, tuple[object, object]] = {}
    expected_method_names: frozenset[str] | None = None
    grouped: dict[
        tuple[str, Digest],
        dict[tuple[DatasetId, MatrixId], list[PreparedComparisonExecution]],
    ] = {}
    for item in pending:
        if not isinstance(item, PreparedComparisonExecution):
            continue
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


def _load_one_matrix_workload(
    dataset: DatasetId,
    matrix_id: MatrixId,
    systems: tuple[PreparedComparisonExecution, ...],
) -> MatrixWorkload:
    """Build one bundle in a helper scope so producer temporaries die before the next load."""
    first_prepared = systems[0]
    first_cfg = first_prepared.cfg
    first_resolved = resolve_comparison_input(
        matrix_path=Path(first_cfg.general.data.matrix_path),
        matrix_dataset_id=first_prepared.entry.matrix_dataset,
        matrix_index=first_cfg.general.data.matrix_index,
        matrix_provenance=first_prepared.planned.provenance,
        require_non_residual_rhs=first_cfg.general.data.require_non_residual_rhs,
        seed=first_cfg.general.data.selection_seed,
        rhs_source_kind=_require_rhs_source_kind(first_cfg),
        rhs_source_params=first_cfg.general.data.rhs_source_params,
    )
    first_system = prepare_resolved_linear_system(
        first_resolved, first_cfg.general.data.normalize_system
    )
    members = [
        MatrixWorkloadMember(
            tag=system_tag(first_prepared.planned),
            rhs=first_system.rhs.numpy(),
        )
    ]
    for prepared in systems[1:]:
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
        members.append(
            MatrixWorkloadMember(
                tag=system_tag(prepared.planned),
                rhs=system.rhs.numpy(),
            )
        )
    return MatrixWorkload(
        dataset=dataset,
        matrix_id=matrix_id,
        matrix=first_system.matrix.numpy(),
        matrix_path=Path(first_cfg.general.data.matrix_path),
        matrix_index=first_cfg.general.data.matrix_index,
        members=tuple(members),
    )


def _load_matrix_workloads(
    matrices: Sequence[
        tuple[
            tuple[DatasetId, MatrixId],
            tuple[PreparedComparisonExecution, ...],
        ]
    ],
) -> Iterator[MatrixWorkload]:
    """Yield one method matrix/RHS bundle at a time before online timing begins."""
    for (dataset, matrix_id), systems in matrices:
        yield _load_one_matrix_workload(dataset, matrix_id, systems)


def _outcomes_from_workloads(
    pending: Sequence[PendingComparison],
    workloads: Sequence[CompletedMethodWorkload],
) -> list[ComparisonOutcome]:
    """Project canonical diagnostics back into the retained per-system API."""
    diagnostics: dict[str, dict[str, CGComparisonResult]] = {}
    for workload in workloads:
        for system_id, result in workload.diagnostics:
            result = _project_setup_measurements(workload, system_id, result)
            diagnostics.setdefault(str(system_id), {})[str(workload.method)] = result

    outcomes: list[ComparisonOutcome] = []
    for item in pending:
        if not isinstance(item, PreparedComparisonExecution):
            outcomes.append(item)
            continue
        results = diagnostics.get(item.entry.id, {})
        if not results:
            outcomes.append(
                _described_failure(item, RuntimeError("method workload produced no diagnostics"))
            )
            continue
        try:
            resolved = resolve_comparison_input(
                matrix_path=Path(item.cfg.general.data.matrix_path),
                matrix_dataset_id=item.entry.matrix_dataset,
                matrix_index=item.cfg.general.data.matrix_index,
                matrix_provenance=item.planned.provenance,
                require_non_residual_rhs=item.cfg.general.data.require_non_residual_rhs,
                seed=item.cfg.general.data.selection_seed,
                rhs_source_kind=_require_rhs_source_kind(item.cfg),
                rhs_source_params=item.cfg.general.data.rhs_source_params,
            )
            system = prepare_resolved_linear_system(
                resolved, item.cfg.general.data.normalize_system
            )
            resolved = replace(
                resolved,
                matrix=system.matrix.numpy(),
                rhs=system.rhs.numpy(),
            )
            with tempfile.TemporaryDirectory() as projection_root:
                result = build_comparison_result_from_diagnostics(
                    general_params=item.cfg.general,
                    preconditioner_configs=item.resolved.specs,
                    resolved_input=resolved,
                    results=dict(results),
                    output_root=Path(projection_root),
                    presentation=item.comparison_presentation,
                )
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            outcomes.append(_described_failure(item, exc))
            continue
        outcomes.append(
            ComparisonCompleted(
                comparison_id=item.entry.id,
                run=item.run_name,
                warnings=item.resolved.warnings,
                payload=result,
            )
        )
    return outcomes


def _project_setup_measurements(
    workload: CompletedMethodWorkload,
    system_id: str,
    result: CGComparisonResult,
) -> CGComparisonResult:
    """Copy the linked setup event into a temporary legacy result projection."""
    tag = next(tag for tag in workload.planned if str(tag.system_id) == system_id)
    setup_event = next(
        event
        for event in workload.setup_events
        if event.dataset == tag.dataset and event.matrix_id == tag.matrix_id
    )
    if not isinstance(setup_event.outcome, SetupCompleted):
        return result
    return replace(
        result,
        setup_cost=setup_event.setup_s,
        setup_time_samples=setup_event.samples_s,
        setup_peak_memory_bytes=(
            setup_event.peak_memory.bytes
            if isinstance(setup_event.peak_memory, MeasuredPeakMemory)
            else None
        ),
    )


@contextlib.contextmanager
def _failure_boundary(
    sink: WorkflowFailureSink,
    context: WorkflowFailureContext,
    *,
    recorded_exception_ids: set[int] | None = None,
) -> Generator[None]:
    """Record a blocking failure while preserving its exact exception and traceback."""
    try:
        yield
    except Exception as exc:
        exc.add_note(f"Comparison workflow context: {context!r}")
        exception_id = id(exc)
        if recorded_exception_ids is not None and exception_id in recorded_exception_ids:
            raise
        if recorded_exception_ids is not None:
            recorded_exception_ids.add(exception_id)
        try:
            sink.record(context, exc)
        except Exception as diagnostic_exc:  # noqa: BLE001
            logger.warning(
                "Failure diagnostics raised while preserving '{}': {}",
                exc,
                diagnostic_exc,
            )
        raise


def _load_matrix_workloads_with_failure_context(
    method: ResolvedMethod,
    matrices: Sequence[
        tuple[
            tuple[DatasetId, MatrixId],
            tuple[PreparedComparisonExecution, ...],
        ]
    ],
    *,
    failure_sink: WorkflowFailureSink,
    recorded_exception_ids: set[int],
) -> Iterator[MatrixWorkload]:
    """Load lazily while attributing a blocking read to its exact matrix."""
    for (dataset, matrix_id), systems in matrices:
        with _failure_boundary(
            failure_sink,
            ComparisonMatrixFailureContext(
                phase=ComparisonMatrixPhase.INPUT_LOADING,
                method=method.name,
                dataset=dataset,
                matrix_id=matrix_id,
            ),
            recorded_exception_ids=recorded_exception_ids,
        ):
            bundle = next(iter(_load_matrix_workloads((((dataset, matrix_id), systems),))))
        yield bundle


def _execute_method_run(
    method: ResolvedMethod,
    matrices: tuple[
        tuple[tuple[DatasetId, MatrixId], tuple[PreparedComparisonExecution, ...]], ...
    ],
    *,
    method_sink: BenchmarkMethodSink,
    failure_sink: WorkflowFailureSink,
) -> CompletedMethodWorkload:
    """Execute and persist one method inside the method's own MLflow child."""
    system_ids = tuple(str(system.entry.id) for _, systems in matrices for system in systems)
    system_identities = tuple(system.identity.key for _, systems in matrices for system in systems)
    config_digest = method.identity_digest or canonical_digest(method.config)
    # Using the first execution's prepared method is safe: the schedule groups
    # executions by method name and identity digest and rejects differing
    # identities or method membership, so every execution here configures the
    # same method over the same cohort and prepares an equal presentation.
    first = matrices[0][1][0]
    prepared_method = first.presentation.method(method.name)
    run, tags = build_benchmark_method_run_spec(
        prepared=prepared_method,
        notation=first.cfg.notation_policy,
        config_digest=config_digest,
        system_ids=system_ids,
        system_identities=system_identities,
    )
    context_fields = {
        "method": method.name,
        "config_digest": config_digest,
        "system_ids": tuple(SystemId(system_id) for system_id in system_ids),
    }
    recorded_exception_ids: set[int] = set()
    with mlflow.start_run(run_name=run_name_text(run), nested=True, tags=tags.as_mlflow_tags()):
        log_configured_method_facts(prepared_method)
        with _failure_boundary(
            failure_sink,
            ComparisonMethodFailureContext(
                phase=ComparisonMethodPhase.EXECUTION,
                **context_fields,
            ),
            recorded_exception_ids=recorded_exception_ids,
        ):
            workload = execute_method_workload(
                method,
                _load_matrix_workloads_with_failure_context(
                    method,
                    matrices,
                    failure_sink=failure_sink,
                    recorded_exception_ids=recorded_exception_ids,
                ),
                prepared=prepared_method,
                datasets={
                    dataset: first_of_dataset.presentation.scope.dataset
                    for (dataset, _), systems in matrices
                    for first_of_dataset in systems[:1]
                },
                notation=first.cfg.notation_policy,
                params=first.cfg.general.params,
                settings=first.settings,
            )
            workload = replace(workload, workload_identities=system_identities)
        with _failure_boundary(
            failure_sink,
            ComparisonMethodFailureContext(
                phase=ComparisonMethodPhase.PERSISTENCE,
                **context_fields,
            ),
            recorded_exception_ids=recorded_exception_ids,
        ):
            summary = summarize_method_workload(
                workload.method,
                workload.planned,
                workload.setup_events,
                workload.solve_attempts,
                workload.workload_peak_memory,
            )
            method_sink.record_method(workload, summary)
    return workload


def _execute_all(
    batch: PreparedComparisonBatch,
    *,
    method_sink: BenchmarkMethodSink,
    failure_sink: WorkflowFailureSink,
) -> tuple[CompletedMethodWorkload, ...]:
    """Execute a prepared batch in method order, each method inside its persisted child run."""
    schedule = _method_major_schedule(batch.items)

    workloads: list[CompletedMethodWorkload] = []
    for method, matrices in schedule:
        try:
            workloads.append(
                _execute_method_run(
                    method,
                    matrices,
                    method_sink=method_sink,
                    failure_sink=failure_sink,
                )
            )
        finally:
            release_device_memory()
    return tuple(workloads)


def _has_complete_reusable_session(
    batch: PreparedComparisonBatch,
    topology: ComparisonTopology,
    *,
    force: bool,
) -> bool:
    """Require current method children under one finished parent with a valid manifest."""
    if force:
        return False
    schedule = _method_major_schedule(batch.items)
    if not schedule:
        return False
    client = MlflowClient(tracking_uri=topology.tracking_uri)
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
    if parent.info.status != "FINISHED" or parent.data.tags.get("phase") != "session_comparison":
        return False
    manifest_artifact = f"{GROUP_REPORT_ARTIFACT_DIR}/figure_manifest.toml"
    if not any(
        artifact.path == manifest_artifact
        for artifact in client.list_artifacts(parent_id, path=GROUP_REPORT_ARTIFACT_DIR)
    ):
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="neuralls-session-reuse-") as temp_dir:
            manifest_path = client.download_artifacts(parent_id, manifest_artifact, temp_dir)
            manifest = tomllib.loads(Path(manifest_path).read_text(encoding="utf-8"))
        validate_figure_manifest(manifest)
    except OSError, tomllib.TOMLDecodeError, InvalidFigureManifestError:
        return False
    return True


def run_comparison_batch(
    case_config_path: Path,
    params: ComparisonParams,
    settings: NeurallsSettings | None = None,
) -> list[ComparisonOutcome]:
    """Run all configured comparison entries from the case config.

    Every entry is resolved and reuse-checked first, with no MLflow run
    opened yet. Only if at least one entry genuinely needs execution does
    this open one MLflow session parent run for the whole batch — named
    ``"<case-name> | <timestamp>"``, mirroring training's session parent —
    with one direct method child per executing workload. If every entry is a
    cache hit, no MLflow run is created at all.

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

    with contextlib.ExitStack() as stack:
        stack.enter_context(digest_memo())
        batch = prepare_comparison_batch(
            master_cfg,
            topology,
            case_config_path,
            settings,
            mlflow_client,
            stack,
            force=params.force,
            reuse_comparisons=False,
        )
        pending = batch.items

        if not any(isinstance(p, PreparedComparisonExecution) for p in pending):
            outcomes = [p for p in pending if is_comparison_outcome(p)]
            return outcomes

        if _has_complete_reusable_session(batch, topology, force=params.force):
            return [
                item
                if is_comparison_outcome(item)
                else ComparisonReused(
                    comparison_id=item.entry.id,
                    run=item.run_name,
                )
                for item in pending
            ]

        notation = _batch_notation_policy(pending)
        session_run, session_tags = build_session_run_spec(
            title=session_title(
                [item.title for item in pending if isinstance(item, PreparedComparisonExecution)]
            ),
            notation=notation,
            case_config_path=case_config_path.resolve(),
            experiment_name=topology.experiment_name,
            phase="session_comparison",
        )
        failure_sink = MLflowWorkflowFailureSink()
        with mlflow.start_run(
            run_name=run_name_text(session_run), tags=session_tags.as_mlflow_tags()
        ):
            with _failure_boundary(
                failure_sink,
                ComparisonBatchFailureContext(
                    phase=ComparisonBatchPhase.WARMUP,
                    case_config_path=case_config_path.resolve(),
                ),
            ):
                warm_up_device(resolve_device())
            with _failure_boundary(
                failure_sink,
                ComparisonBatchFailureContext(
                    phase=ComparisonBatchPhase.METHOD_WORKLOADS,
                    case_config_path=case_config_path.resolve(),
                ),
            ):
                workloads = _execute_all(
                    batch,
                    method_sink=MLflowBenchmarkMethodSink(
                        notation, _batch_dataset_presentations(pending)
                    ),
                    failure_sink=failure_sink,
                )
            with _failure_boundary(
                failure_sink,
                ComparisonBatchFailureContext(
                    phase=ComparisonBatchPhase.RESULT_PROJECTION,
                    case_config_path=case_config_path.resolve(),
                ),
            ):
                outcomes = _outcomes_from_workloads(pending, workloads)
            selection = _batch_reference_selections(pending)
            contexts_by_system = _batch_solver_timing_contexts(pending)
            report_sink = MLflowGroupReportSink(notation)
            with _failure_boundary(
                failure_sink,
                ComparisonBatchFailureContext(
                    phase=ComparisonBatchPhase.REPORTING,
                    case_config_path=case_config_path.resolve(),
                ),
            ):
                report_sink.register_method_figures(
                    workloads, contexts_by_system=contexts_by_system
                )
                report_all_levels(
                    workloads,
                    report_sink,
                    datasets=_batch_dataset_presentations(pending),
                    notation=notation,
                    reference_selection=selection,
                    contexts_by_system=contexts_by_system,
                )
            with _failure_boundary(
                failure_sink,
                ComparisonBatchFailureContext(
                    phase=ComparisonBatchPhase.FINALIZATION,
                    case_config_path=case_config_path.resolve(),
                ),
            ):
                report_sink.finalize()
            return outcomes


def _batch_notation_policy(pending: Sequence[PendingComparison]) -> NotationPolicy:
    """The one notation every executed comparison of the batch uses; differing ones are ambiguous."""
    policies = [
        item.cfg.notation_policy
        for item in pending
        if isinstance(item, PreparedComparisonExecution)
    ]
    first, *rest = policies
    if any(policy != first for policy in rest):
        raise ValueError("every comparison of a batch must share one notation policy")
    return first


def _batch_dataset_presentations(
    pending: Sequence[PendingComparison],
) -> dict[DatasetId, DatasetPresentation]:
    """The presentation of each matrix dataset the batch runs on, from its manifest."""
    datasets: dict[DatasetId, DatasetPresentation] = {}
    for item in pending:
        if not isinstance(item, PreparedComparisonExecution):
            continue
        dataset = DatasetId(item.entry.matrix_dataset)
        presented = item.presentation.scope.dataset
        if datasets.setdefault(dataset, presented) != presented:
            raise ValueError("one dataset must have one scientific presentation in a batch")
    return datasets


def _batch_reference_selections(
    pending: Sequence[PendingComparison],
) -> dict[SystemId, ReferenceSelection]:
    """Return explicit reference state for each planned system."""
    return {
        SystemId(str(tag.system_id)): item.reference_selection
        for item in pending
        if isinstance(item, PreparedComparisonExecution)
        for tag in (system_tag(item.planned),)
    }


def _batch_solver_timing_contexts(
    pending: Sequence[PendingComparison],
) -> dict[SystemId, SolverTimingContext]:
    """Preserve the actual per-system solver and timing settings for figure metadata."""
    contexts: dict[SystemId, SolverTimingContext] = {}
    for item in pending:
        if not isinstance(item, PreparedComparisonExecution):
            continue
        params = item.cfg.general.params
        contexts[SystemId(str(system_tag(item.planned).system_id))] = SolverTimingContext(
            rtol=params.rtol,
            atol=params.atol,
            max_iterations=params.max_iterations,
            stopping_criterion=params.stopping_criterion,
            m_max=params.m_max,
            reference_precision_margin=params.reference_precision_margin,
            timing_repeats=params.timing.repeats,
            timing_warmup=params.timing.warmup,
            timing_statistic=params.timing.statistic.value,
            normalization=item.cfg.general.data.normalize_system,
        )
    return contexts
