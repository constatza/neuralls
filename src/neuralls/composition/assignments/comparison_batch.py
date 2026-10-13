"""Backend helpers for comparison workflows."""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
import tomli_w
from dlkit.infrastructure.config.job_config import FitJobConfig
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.composition.assignments.job_loader import load_experiment_job
from neuralls.composition.assignments.model_resolution import (
    AssignmentModelContext,
    TrainingLookup,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.composition.comparison._generation_cost import resolve_generation_repeatability
from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison._preconditioner_setup import SweepPreconditionerCache
from neuralls.composition.comparison.comparison_run import compare_preconditioners
from neuralls.composition.comparison.config_assembler import (
    resolve_comparison_config,
    resolve_dataset_dir,
)
from neuralls.composition.comparison.models import (
    ComparisonGroup,
    ComparisonGroupings,
    ComparisonOutcome,
    ComparisonParams,
    GroupMatrixIdentity,
    MultiMatrixDataset,
    SingleMatrixDataset,
)
from neuralls.composition.comparison.pinned_split import (
    resolve_test_matrix_indices,
    resolve_test_row_indices,
)
from neuralls.composition.comparison.result_keys import validate_unique_preconditioner_keys
from neuralls.composition.comparison.source_handlers import (
    ComparisonSourceSpec,
    validate_comparison_source,
)
from neuralls.composition.comparison.test_set_sweep import expand_test_set_sweep
from neuralls.composition.identity.comparison import comparison_identity
from neuralls.composition.identity.gate import gate_reuse
from neuralls.composition.identity.training import training_identity
from neuralls.composition.tracking.run_specs import (
    ComparisonRunTags,
    build_child_comparison_tags,
    build_comparison_run_spec,
    build_session_run_spec,
)
from neuralls.domain.identity import StageIdentity
from neuralls.domain.solver.comparison import classify_group_aggregation
from neuralls.domain.solver.models.provenance import PinnedSplitMatrix, UnverifiedMatrix
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonResult,
    GroupAggregated,
    GroupSkippedAllCacheHit,
    ProjectedComparisonResult,
    SetupSharingPolicy,
)
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
from neuralls.platform.config.registry import get_assignment_binding, resolve_assignment_binding
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.artifacts import write_comparison_artifacts
from neuralls.platform.reporting.comparison_inputs import (
    ComparisonInputArtifacts,
    stage_comparison_inputs,
)
from neuralls.platform.reporting.plots import plot_projected_scaling_law
from neuralls.platform.storage.dataset_readers import list_available_matrix_indices
from neuralls.platform.tracking.artifact_access import (
    ArtifactLeaseManager,
    MlflowArtifactLeaseManager,
)
from neuralls.platform.tracking.comparison_tracking import (
    create_cost_recorder,
    log_comparison_artifact_uri,
    log_comparison_input_artifacts,
    log_comparison_run_params,
    log_linear_system_params,
    log_projected_metrics,
    log_skipped_preconditioners,
    setup_comparison_tracking,
)
from neuralls.platform.tracking.extra_features import fetch_extra_feature_names
from neuralls.platform.tracking.mlflow import build_workflow_environment
from neuralls.platform.tracking.mlflow_client import log_comparison_artifacts_to_mlflow
from neuralls.platform.tracking.mlflow_store import MlflowIdentityStore
from neuralls.platform.tracking.model_registry import build_registered_model_name
from neuralls.shared.digest import Digest, content_digest
from neuralls.shared.types import (
    ComparisonRhsSourceKind,
    DatasetId,
    GenerationRepeatability,
    MatrixId,
    TestPosition,
)


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
    """Derive an assignment's training identity, or ``None`` if its dataset is not generated.

    ``None`` makes strict model resolution fail for that assignment with a
    clear message (never a silent match against an unverifiable dataset).
    """
    try:
        return training_identity(
            job_config_path=job_config_path,
            data_config_path=data_config_path,
            settings=settings,
        )
    except FileNotFoundError:
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
    """

    topology: ComparisonTopology
    case_config_path: Path
    settings: NeurallsSettings
    master_cfg: CaseConfig | None = None


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
    entry: ComparisonRegistryEntry,
    work_root: Path,
    resolved_specs: list[PreconditionerConfig],
    settings: NeurallsSettings,
    cache: SweepPreconditionerCache | None = None,
) -> ComparisonResult:
    """Run comparison with already-resolved preconditioner checkpoint paths.

    `cache`, when given, lets a preconditioner built for an earlier sibling
    system in the same `comparison_sweeps` group be reused here instead of
    rebuilt — see `SweepPreconditionerCache`. `None` (the default) means
    this call builds every preconditioner fresh, unchanged from before this
    parameter existed.
    """
    rhs_source_kind = _require_rhs_source_kind(cfg)
    matrix_path = Path(cfg.general.data.matrix_path)
    test_position = cfg.general.data.test_position
    matrix_index = cfg.general.data.matrix_index
    rhs_source_params = cfg.general.data.rhs_source_params
    if test_position is not None:
        if rhs_source_kind is ComparisonRhsSourceKind.DATASET:
            test_rows = resolve_test_row_indices(matrix_path)
            if test_position >= len(test_rows):
                raise IndexError(
                    f"test_position={test_position} is out of range for dataset "
                    f"'{matrix_path}', which has {len(test_rows)} test-fold rows."
                )
            resolved_row = test_rows[test_position]
            rhs_source_params = {**(rhs_source_params or {}), "sample_index": resolved_row}
        else:
            test_matrix_indices = resolve_test_matrix_indices(matrix_path)
            if test_position >= len(test_matrix_indices):
                raise IndexError(
                    f"test_position={test_position} is out of range for dataset "
                    f"'{matrix_path}', which has {len(test_matrix_indices)} test-fold matrices."
                )
            matrix_index = test_matrix_indices[test_position]

    resolved_input = resolve_comparison_input(
        matrix_path=matrix_path,
        matrix_dataset_id=entry.matrix_dataset,
        matrix_index=matrix_index,
        require_non_residual_rhs=cfg.general.data.require_non_residual_rhs,
        seed=cfg.general.data.selection_seed,
        rhs_source_kind=rhs_source_kind,
        rhs_source_params=rhs_source_params,
    )
    if resolved_input.matrix_provenance is None:
        # Source handlers that know their own provenance (e.g. cross-dataset)
        # set it directly; every other kind is pinned-split or unverified
        # depending only on whether a test_position was requested here.
        if test_position is not None:
            matrix_provenance = PinnedSplitMatrix(
                test_position=TestPosition(test_position),
                raw_matrix_index=MatrixId(resolved_input.matrix_index),
            )
        else:
            matrix_provenance = UnverifiedMatrix(
                raw_matrix_index=MatrixId(resolved_input.matrix_index)
            )
        resolved_input = replace(resolved_input, matrix_provenance=matrix_provenance)
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
    cache: SweepPreconditionerCache | None = None,
) -> ComparisonResult:
    """Run one comparison in a scratch directory, logging into the open MLflow run.

    Args:
        prepared: The entry, config, topology, and resolved specs to execute.
        run_name: Display name of the enclosing run, used as the child runs' parent label.
        comp_run_id: Id of the enclosing run that artifacts are uploaded to.
        cost_recorder: Where to record the comparison's cost/outcome metrics
            (DI for testing); defaults to MLflow, logging into the open run.
        cache: Optional per-sweep-group preconditioner cache (see
            `SweepPreconditionerCache`); `None` for a non-sweep entry.

    Returns:
        The comparison result, after its artifacts and metrics have been logged.
    """
    recorder = cost_recorder if cost_recorder is not None else create_cost_recorder()
    with tempfile.TemporaryDirectory() as tmp:
        work_root = Path(tmp)
        raw_result = _run_comparison_with_resolved_specs(
            cfg=prepared.cfg,
            entry=prepared.entry,
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
    cache: SweepPreconditionerCache | None = None,
) -> ComparisonOutcome:
    """Open an MLflow run, execute the comparison, upload artifacts, and report the outcome.

    Args:
        prepared: Everything resolved ahead of execution — config, registry
            entry, topology, resolved specs, warnings, and dependency hash.
        run_name: Display name for the MLflow run.
        comp_tags: Structured tags applied to the MLflow run.
        cache: Optional per-sweep-group preconditioner cache (see
            `SweepPreconditionerCache`); `None` for a non-sweep entry.

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
    logger.error(f"Comparison failed: {exc}")
    return _comparison_outcome(entry, success=False, error=str(exc))


@dataclass(frozen=True)
class _PreparedComparisonExecution:
    """One entry, resolved and confirmed to need real execution.

    `cleanup` owns the entry's artifact-lease lifetime (already resolved into
    `resolved.specs`' baked-in checkpoint paths) — the caller must enter it
    into its own ExitStack (or close it directly) once execution is done.
    """

    cfg: ComparisonConfig
    entry: ComparisonRegistryEntry
    topology: ComparisonTopology
    cleanup: contextlib.ExitStack
    resolved: ResolvedComparisonSpecs
    identity: StageIdentity
    settings: NeurallsSettings


def _prepare_comparison_entry(
    cfg: ComparisonConfig,
    entry: ComparisonRegistryEntry,
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

    Every terminal outcome (validation failure, resolution failure, or a
    cache hit) closes the entry's artifact lease manager before returning. A
    `_PreparedComparisonExecution` instead detaches that lifetime for the
    caller to keep open through execution.

    Args:
        cfg: Fully resolved ComparisonConfig with injected data paths and preconditioners.
        entry: Registry entry providing display name and method path for artifact logging.
        context: Shared resolution context (topology, case config, settings).
        force: Treat this entry as needing execution even if a matching comparison
            already completed successfully.

    Returns:
        A cache-hit/failure `ComparisonOutcome`, or a `_PreparedComparisonExecution`
        ready for `_execute_prepared_comparison`.
    """
    logger.info("Preparing comparison '{}': validating input sources.", entry.id)
    try:
        _validate_comparison_sources(cfg)
    except (FileNotFoundError, ValueError) as exc:
        return _comparison_outcome(entry, success=False, error=str(exc))
    logger.info("Comparison '{}': input-source validation complete.", entry.id)

    model_client = MlflowClient(tracking_uri=context.topology.model_store_tracking_uri)
    with contextlib.ExitStack() as local_stack:
        artifact_leases = local_stack.enter_context(MlflowArtifactLeaseManager(client=model_client))
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
            entry=entry,
            topology=context.topology,
            cleanup=local_stack.pop_all(),
            resolved=resolved,
            identity=identity,
            settings=context.settings,
        )


def _execute_prepared_comparison(
    prepared: _PreparedComparisonExecution,
    *,
    cache: SweepPreconditionerCache | None = None,
) -> ComparisonOutcome:
    """Build the run spec/tags and execute inside an already-active nested MLflow run.

    Args:
        prepared: Everything resolved ahead of execution.
        cache: Optional per-sweep-group preconditioner cache (see
            `SweepPreconditionerCache`); `None` for a non-sweep entry.
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
    entry: ComparisonRegistryEntry,
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
        entry: Registry entry providing display name and method path for artifact logging.
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
    prepared = _prepare_comparison_entry(cfg, entry, context, force=force)
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


def _expand_comparison_sweeps(
    master_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> list[ComparisonRegistryEntry]:
    """Expand every `[[comparison_sweeps]]` entry into concrete comparison entries.

    Each expanded entry flows into `_prepare_all_comparisons` indistinguishable
    from a hand-written `[[comparisons]]` entry with `test_position` set.
    Which comparison-group it belongs to (if any) is discovered later, from
    data, by `_discover_comparison_groups` — identically for this entry and
    for any hand-written one.
    """
    expanded: list[ComparisonRegistryEntry] = []
    for sweep_entry in master_cfg.comparison_sweeps:
        dataset_dir = resolve_dataset_dir(
            master_cfg, config_dir, sweep_entry.matrix_dataset, settings
        )
        expanded.extend(expand_test_set_sweep(sweep_entry, dataset_dir=dataset_dir))
    return expanded


def _group_key_by_matrix(entry: ComparisonRegistryEntry) -> str:
    """Partition key for execution-time setup-sharing and the coarse report.

    `rhs_source.kind` plays no part: a preconditioner's `setup()` never
    sees the RHS, so whether its build can be reused across entries is a
    question about `matrix_dataset` alone.
    """
    return entry.matrix_dataset


def _group_key_by_matrix_and_rhs_kind(entry: ComparisonRegistryEntry) -> str:
    """Partition key for the finer, reporting-only aggregation.

    Keeps same-matrix entries apart when they differ in `rhs_source.kind`,
    so a preconditioner's cost/iteration behavior under one forcing
    structure (gaussian, sparse, sinusoidal, ...) isn't blended into
    another's average.
    """
    return f"{entry.matrix_dataset}::{entry.rhs_source.kind}"


def _resolve_entry_matrix_id(entry: ComparisonRegistryEntry, dataset_dir: Path) -> MatrixId:
    """Resolve which matrix one entry addresses, without executing it.

    Mirrors `_run_comparison_with_resolved_specs`'s own resolution (needed
    there for RHS/LHS loading too) — this only needs the matrix identity,
    run cheaply and early for group discovery, since both paths are pure
    reads of the dataset's pinned split / matrix-sample-index metadata.
    """
    if entry.test_position is None:
        return MatrixId(entry.matrix_index)
    if entry.rhs_source.kind == ComparisonRhsSourceKind.DATASET:
        # `test_position` here only selects which RHS row to use; the
        # dataset is single-matrix by this kind's own precondition, so
        # `matrix_index` (unchanged, default 0) is already correct.
        return MatrixId(entry.matrix_index)
    test_matrix_indices = resolve_test_matrix_indices(dataset_dir)
    if entry.test_position >= len(test_matrix_indices):
        raise IndexError(
            f"test_position={entry.test_position} is out of range for dataset "
            f"'{dataset_dir}', which has {len(test_matrix_indices)} test-fold matrices."
        )
    return MatrixId(test_matrix_indices[entry.test_position])


def _discover_comparison_groups(
    all_entries: Sequence[ComparisonRegistryEntry],
    master_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
    *,
    key_fn: Callable[[ComparisonRegistryEntry], str],
) -> Mapping[str, ComparisonGroup]:
    """Group every resolved entry by `key_fn`, from data.

    One algorithm, pluggable partition key (`_group_key_by_matrix` or
    `_group_key_by_matrix_and_rhs_kind` — see `_discover_comparison_groupings`,
    which calls this once per key to build both views of one batch).
    Hand-written `[[comparisons]]` entries and `comparison_sweeps`-expanded
    entries are grouped identically regardless of which key is used — group
    membership is a property of what an entry tests, never of which TOML
    array declared it. A group's `GroupMatrixIdentity` is resolved from the
    dataset and each member's actual matrix index, never inferred from
    `rhs_source.kind`.
    """
    members_by_key: dict[str, list[ComparisonRegistryEntry]] = {}
    for entry in all_entries:
        members_by_key.setdefault(key_fn(entry), []).append(entry)

    groups: dict[str, ComparisonGroup] = {}
    for key, members in members_by_key.items():
        dataset_dir = resolve_dataset_dir(
            master_cfg, config_dir, members[0].matrix_dataset, settings
        )
        dataset_id = DatasetId(members[0].matrix_dataset)
        identity: GroupMatrixIdentity
        if len(list_available_matrix_indices(dataset_dir)) <= 1:
            identity = SingleMatrixDataset(dataset_id=dataset_id)
        else:
            identity = MultiMatrixDataset(
                dataset_id=dataset_id,
                resolved_indices=frozenset(
                    _resolve_entry_matrix_id(member, dataset_dir) for member in members
                ),
            )
        groups[key] = ComparisonGroup(
            group_id=key,
            matrix_identity=identity,
            entry_ids=frozenset(member.id for member in members),
        )
    return groups


def _discover_comparison_groupings(
    all_entries: Sequence[ComparisonRegistryEntry],
    master_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> ComparisonGroupings:
    """Discover both grouping granularities `run_comparison_batch` needs.

    `by_matrix` backs `SweepPreconditionerCache`; `by_matrix_and_rhs_kind`
    is reported alongside it, never used for caching — see
    `ComparisonGroupings`'s own docstring for why the two differ.
    """
    return ComparisonGroupings(
        by_matrix=_discover_comparison_groups(
            all_entries, master_cfg, config_dir, settings, key_fn=_group_key_by_matrix
        ),
        by_matrix_and_rhs_kind=_discover_comparison_groups(
            all_entries,
            master_cfg,
            config_dir,
            settings,
            key_fn=_group_key_by_matrix_and_rhs_kind,
        ),
    )


def _group_for_entry(
    groups: Mapping[str, ComparisonGroup], entry_id: str
) -> ComparisonGroup | None:
    """The one group `entry_id` belongs to, or `None` if it belongs to none."""
    return next((group for group in groups.values() if entry_id in group.entry_ids), None)


def _cache_for_group(
    group: ComparisonGroup,
    by_matrix: Mapping[str, ComparisonGroup],
    group_caches: Mapping[str, SweepPreconditionerCache],
) -> SweepPreconditionerCache | None:
    """The execution-time cache covering `group`'s members.

    Always resolved via the coarse (`by_matrix`) grouping, regardless of
    which grouping `group` itself came from — `SweepPreconditionerCache`
    is only ever scoped to `by_matrix` groups (§ `ComparisonGroupings`), so
    a `by_matrix_and_rhs_kind` group (a subset of exactly one `by_matrix`
    group, since matching both keys implies matching the first) must look
    its cache up by proxy through any one of its own members.
    """
    representative_entry_id = next(iter(group.entry_ids), None)
    if representative_entry_id is None:
        return None
    coarse_group = _group_for_entry(by_matrix, representative_entry_id)
    if coarse_group is None:
        return None
    return group_caches.get(coarse_group.group_id)


def _validate_one_matrix_shape_per_case(
    all_entries: Sequence[ComparisonRegistryEntry],
    master_cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> None:
    """Reject a case that mixes single-matrix and multi-matrix dataset references.

    A case is one coherent experimental question — a multi-matrix case and
    a single-matrix one answer genuinely different questions and produce
    differently-shaped plot evidence — so mixing the two dataset shapes in
    one case file is rejected outright rather than left to authoring
    discipline.
    """
    is_multi_by_dataset: dict[str, bool] = {}
    for entry in all_entries:
        if entry.matrix_dataset in is_multi_by_dataset:
            continue
        dataset_dir = resolve_dataset_dir(master_cfg, config_dir, entry.matrix_dataset, settings)
        is_multi_by_dataset[entry.matrix_dataset] = (
            len(list_available_matrix_indices(dataset_dir)) > 1
        )
    if len(set(is_multi_by_dataset.values())) <= 1:
        return
    single = sorted(name for name, is_multi in is_multi_by_dataset.items() if not is_multi)
    multi = sorted(name for name, is_multi in is_multi_by_dataset.items() if is_multi)
    raise ValueError(
        "Case config mixes single-matrix and multi-matrix dataset references "
        f"(single-matrix: {single}; multi-matrix: {multi}). A case must be one "
        "matrix shape — split this into separate case files."
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
) -> tuple[list[ComparisonOutcome | _PreparedComparisonExecution], ComparisonGroupings]:
    """Resolve + reuse-check every [[comparisons]]/[[comparison_sweeps]] entry.

    Opens no MLflow run. Every `[[comparison_sweeps]]` entry is expanded into
    concrete `ComparisonRegistryEntry` objects first (`_expand_comparison_sweeps`),
    so sweep-expanded entries are resolved and reuse-checked exactly like any
    hand-written `[[comparisons]]` entry. Entries needing execution have their
    cleanup registered onto `stack`, which the caller must keep open until
    after execution.

    Returns:
        One outcome-or-prepared-execution per entry, in declared config
        order, plus both `ComparisonGroupings` discovered across all of them.
    """
    config_dir = case_config_path.resolve().parent
    context = _ComparisonResolutionContext(
        topology=topology,
        case_config_path=case_config_path,
        settings=settings,
        master_cfg=master_cfg,
    )
    pending: list[ComparisonOutcome | _PreparedComparisonExecution] = []
    expanded_sweep_entries = _expand_comparison_sweeps(master_cfg, config_dir, settings)
    all_entries = [*master_cfg.comparisons, *expanded_sweep_entries]
    _validate_one_matrix_shape_per_case(all_entries, master_cfg, config_dir, settings)
    groupings = _discover_comparison_groupings(all_entries, master_cfg, config_dir, settings)
    for entry in all_entries:
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
        validate_unique_preconditioner_keys(cfg.preconditioners)

        prepared = _prepare_comparison_entry(cfg, entry, context, force=force)
        if isinstance(prepared, _PreparedComparisonExecution):
            stack.enter_context(prepared.cleanup)
        pending.append(prepared)
    return pending, groupings


def _projected_results_toml_payload(
    projected: Mapping[str, ProjectedComparisonResult],
) -> dict[str, Any]:
    """Build one group's scaling-law numbers as a TOML-ready payload.

    Pure scalars per preconditioner — the plotted curve
    (`plot_projected_scaling_law`) is a deterministic function of exactly
    these fields, so this is the source of truth the PNG is derived from,
    not a separate, possibly-drifting summary of it.
    """
    payload: dict[str, Any] = {}
    for name, result in projected.items():
        entry: dict[str, Any] = {
            "n_systems": result.n_systems,
            "setup_cost_total_seconds": result.setup_cost_total_seconds,
            "total_solve_time_seconds": result.total_solve_time_seconds,
            "total_iterations": result.total_iterations,
            "share_setup": result.setup_sharing.share_setup,
            "share_generation": result.setup_sharing.share_generation,
        }
        if result.generation_cost_total_seconds is not None:
            entry["generation_cost_total_seconds"] = result.generation_cost_total_seconds
        payload[name] = entry
    return payload


def _log_group_aggregation(
    outcomes: Sequence[ComparisonOutcome],
    pending: Sequence[ComparisonOutcome | _PreparedComparisonExecution],
    groups: Mapping[str, ComparisonGroup],
    by_matrix: Mapping[str, ComparisonGroup],
    group_caches: Mapping[str, SweepPreconditionerCache],
) -> None:
    """Classify and report every discovered group's aggregation outcome.

    Visits every group discovered for this batch — including one where
    every member was a cache hit — so a fully- or partially-skipped group
    is logged loudly instead of silently producing nothing. Safe to call
    with no open MLflow run as long as every group classifies as
    `GroupSkippedAllCacheHit` (that branch only logs; see
    `run_comparison_batch`'s all-cache-hit short-circuit, which relies on
    this).

    Called once per grouping granularity (see
    `ComparisonGroupings`/`_log_all_groupings`) — `groups` is whichever
    grouping is being reported this call, while `by_matrix` is always the
    coarse grouping, since that's the only one a cache was ever built
    against (`_cache_for_group` resolves through it regardless of `groups`).

    Args:
        outcomes: Every outcome `run_comparison_batch` is about to return.
        pending: The same-indexed prepared batch `outcomes` was built from —
            read for `_PreparedComparisonExecution.resolved.specs`, the
            resolved preconditioner configs a group's members share.
        groups: The `ComparisonGroup`s being classified/reported this call.
        by_matrix: The coarse grouping, for cache resolution only.
        group_caches: Each `by_matrix` group's preconditioner cache, if one
            needed constructing (empty for a fully-cache-hit batch).
    """
    specs_by_entry_id = {
        p.entry.id: p.resolved.specs for p in pending if isinstance(p, _PreparedComparisonExecution)
    }
    outcomes_by_id = {outcome.comparison_id: outcome for outcome in outcomes}

    for group in groups.values():
        results_by_system: dict[str, dict[str, CGComparisonResult]] = {}
        for entry_id in group.entry_ids:
            outcome = outcomes_by_id.get(entry_id)
            if outcome is None or outcome.payload is None:
                continue
            results_by_system[entry_id] = outcome.payload.results
        specs = next(
            (
                specs_by_entry_id[entry_id]
                for entry_id in group.entry_ids
                if entry_id in specs_by_entry_id
            ),
            [],
        )
        repeatability_by_name = {
            spec.name: resolve_generation_repeatability(spec) for spec in specs
        }
        cache = _cache_for_group(group, by_matrix, group_caches)
        shared_names = cache.shared_names() if cache is not None else frozenset[str]()
        sharing_by_name = {
            name: SetupSharingPolicy(
                share_setup=name in shared_names,
                share_generation=repeatability is GenerationRepeatability.ONE_TIME,
            )
            for name, repeatability in repeatability_by_name.items()
        }

        outcome = classify_group_aggregation(
            group.group_id, len(group.entry_ids), results_by_system, sharing_by_name
        )
        match outcome:
            case GroupAggregated(
                group_id=group_id,
                projected=projected,
                n_aggregated=n_aggregated,
                n_in_group=n_in_group,
            ):
                if n_aggregated < n_in_group:
                    logger.warning(
                        "Group '{}': aggregated {} of {} members — the rest were cache hits.",
                        group_id,
                        n_aggregated,
                        n_in_group,
                    )
                log_projected_metrics(group_id, projected)
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_dir = Path(tmp)
                    plot_path = tmp_dir / f"projected_scaling_law_{group_id}.png"
                    plot_projected_scaling_law(
                        projected,
                        save_path=plot_path,
                        title=f"Projected cost per system vs. N — group '{group_id}'",
                    )
                    toml_path = tmp_dir / f"projected_scaling_law_{group_id}.toml"
                    with toml_path.open("wb") as fh:
                        tomli_w.dump(_projected_results_toml_payload(projected), fh)
                    mlflow.log_artifact(str(plot_path), artifact_path="projected_scaling_law")
                    mlflow.log_artifact(str(toml_path), artifact_path="projected_scaling_law")
            case GroupSkippedAllCacheHit(group_id=group_id, n_in_group=n_in_group):
                logger.warning(
                    "Group '{}': all {} members were cache hits — nothing to aggregate this run.",
                    group_id,
                    n_in_group,
                )


def _log_all_groupings(
    outcomes: Sequence[ComparisonOutcome],
    pending: Sequence[ComparisonOutcome | _PreparedComparisonExecution],
    groupings: ComparisonGroupings,
    group_caches: Mapping[str, SweepPreconditionerCache],
) -> None:
    """Report both grouping granularities for one batch.

    Two independent aggregations over the same executed results: by
    matrix alone (what `SweepPreconditionerCache` actually shares setup
    across) and by matrix-and-RHS-kind (keeps same-matrix,
    different-RHS-kind systems apart in the report, without affecting
    what was cached during execution).
    """
    _log_group_aggregation(
        outcomes, pending, groupings.by_matrix, groupings.by_matrix, group_caches
    )
    _log_group_aggregation(
        outcomes, pending, groupings.by_matrix_and_rhs_kind, groupings.by_matrix, group_caches
    )


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
        List of comparison outcomes, one per [[comparisons]] entry plus one per
        position expanded from each [[comparison_sweeps]] entry, in declared
        config order (sweep-expanded entries follow all [[comparisons]] entries).
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
        pending, groupings = _prepare_all_comparisons(
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
            _log_all_groupings(outcomes, pending, groupings, {})
            return outcomes

        with mlflow.start_run(run_name=session_run_name, tags=session_tags.as_mlflow_tags()):
            group_caches: dict[str, SweepPreconditionerCache] = {}
            outcomes: list[ComparisonOutcome] = []
            for p in pending:
                if isinstance(p, _PreparedComparisonExecution):
                    group = _group_for_entry(groupings.by_matrix, p.entry.id)
                    cache = None
                    if group is not None:
                        cache = group_caches.setdefault(
                            group.group_id,
                            SweepPreconditionerCache(matrix_identity=group.matrix_identity),
                        )
                    outcomes.append(_execute_prepared_comparison(p, cache=cache))
                else:
                    outcomes.append(p)
            _log_all_groupings(outcomes, pending, groupings, group_caches)
            return outcomes
