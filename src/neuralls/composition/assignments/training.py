"""Training orchestration helpers consumed by CLI scripts and workflows."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dlkit.engine.workflows.multi_run import RunSpec
from dlkit.infrastructure.config.core.patching import patch_model
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments._dataset_assembly import (
    _extra_feature_names_from_settings,
    _load_and_prepare_data,
    resolve_dataset_format,
)
from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments._settings_pipeline import _configure_training_pipeline
from neuralls.composition.assignments._training_artifacts import (
    _build_training_run_config,
    _get_normalized_training_numpy_payload,
    _log_training_context,
    _log_training_evaluation,
    _resolve_local_training_checkpoint,
    _resolve_mlflow_run_ids,
    _resolve_training_checkpoint,
    _stage_training_artifacts,
    create_fallback_training_run,
)
from neuralls.composition.assignments.assembler import load_assignment
from neuralls.composition.assignments.runtime_dataset_contract import (
    RuntimeDatasetContract,
    default_training_dataset_contract,
)
from neuralls.platform.config.loaders import load_tracking_config
from neuralls.platform.config.models.workspace import AssignmentWorkspace, RunnableAssignment
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.tracking.artifact_access import (
    ArtifactLeaseManager,
    MlflowArtifactLeaseManager,
    NoopArtifactLeaseManager,
)
from neuralls.platform.tracking.extra_features import log_extra_feature_names_tag
from neuralls.platform.tracking.mlflow import (
    MlflowRunConfig,
    MlflowRuntimeEnvironment,
    build_runtime_environment,
    resolve_runtime_tracking_config,
)
from neuralls.platform.tracking.mlflow_client import (
    log_artifacts_to_mlflow,
    mark_run_failed,
)
from neuralls.platform.tracking.model_registry import ensure_checkpoint_artifact


@dataclass(frozen=True)
class TrainingResult:
    """Immutable training result data.

    Attributes:
        checkpoint_path: Path to saved model checkpoint (.ckpt file)
        experiment_dir: Root experiment directory
        data_dir: Directory containing training data
    """

    checkpoint_path: Path
    experiment_dir: Path
    data_dir: Path


def _unwrap_execution_result(result: object) -> object:
    """Normalize DLKit execute() results to the underlying training result."""
    return getattr(result, "training_result", result)


def _training_artifact_lease_scope(
    *,
    tracking_uri: str | None,
    run_id: str | None,
) -> AbstractContextManager[ArtifactLeaseManager]:
    """Return an MLflow-backed lease scope, or a no-op one with no source run."""
    if not tracking_uri or not run_id:
        return NoopArtifactLeaseManager()
    client = MlflowClient(tracking_uri=tracking_uri)
    return MlflowArtifactLeaseManager(client=client)


def _resolve_finalization_checkpoint(
    *,
    context: _TrainingFinalizationContext,
    run_id: str | None,
    artifact_leases: ArtifactLeaseManager,
) -> Path:
    """Resolve a checkpoint for finalization, falling back to MLflow when run_id is known."""
    if run_id is None:
        local_checkpoint = _resolve_local_training_checkpoint(
            training_result=context.training_result,
            workspace=context.workspace,
        )
        if local_checkpoint is not None:
            return local_checkpoint
        raise RuntimeError(f"No checkpoint found in {context.workspace.checkpoint_dir}")
    return _resolve_training_checkpoint(
        training_result=context.training_result,
        workspace=context.workspace,
        run_id=run_id,
        artifact_leases=artifact_leases,
    )


def _finalize_existing_mlflow_run(
    *,
    context: _TrainingFinalizationContext,
    mlflow_coords: tuple[str, str, str],
    checkpoint_path: Path,
) -> tuple[str, str, str]:
    """Make an existing DLKit MLflow run durable."""
    tracking_uri, _, run_id = mlflow_coords
    _log_training_context(
        tracking_uri=tracking_uri,
        run_id=run_id,
        assignment_id=context.assignment.spec.assignment_id,
        assignment_display_name=context.resolved_assignment_display_name,
        resolved_dataset_id=context.dataset_id,
        dataset_display_name=context.resolved_dataset_display_name,
        dataset_registry_id=context.assignment.spec.dataset_id,
        job_registry_id=context.assignment.spec.job_id,
        job_display_name=context.assignment.spec.job_display_name or context.workspace.run_id,
    )
    log_extra_feature_names_tag(
        tracking_uri,
        run_id,
        _extra_feature_names_from_settings(context.workflow_settings, context.contract),
    )
    ensure_checkpoint_artifact(
        tracking_uri=tracking_uri,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
    )
    normalized_numpy = _get_normalized_training_numpy_payload(
        context.training_result,
        context.contract,
    )
    _stage_training_artifacts(
        workspace=context.workspace,
        training_result=context.training_result,
        numpy_payload=normalized_numpy,
        job_config_path=context.config_path,
        data_config_path=context.resolved_data_config_path,
    )
    _log_training_evaluation(
        tracking_uri,
        run_id,
        normalized_numpy,
        context.workspace.figures_dir,
        context.contract,
    )
    log_artifacts_to_mlflow(
        tracking_uri=tracking_uri,
        run_id=run_id,
        workspace_root=context.workspace.root_dir,
    )
    return mlflow_coords


def _finalize_fallback_mlflow_run(
    *,
    context: _TrainingFinalizationContext,
    checkpoint_path: Path,
) -> tuple[str, str, str] | None:
    """Create a fallback MLflow run when DLKit left no run metadata."""
    logger.warning("Training completed without MLflow run metadata; creating fallback run.")
    try:
        fallback_exp_id, fallback_run_id = create_fallback_training_run(
            tracking_uri=context.fallback_tracking_uri or "",
            experiment_name=context.run_config.experiment_name,
            run_name=context.run_config.run_name,
            tags=dict(context.run_config.tags),
            checkpoint_path=checkpoint_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create fallback MLflow run: {}", exc)
        return None
    return (
        context.fallback_tracking_uri or "",
        fallback_exp_id,
        fallback_run_id,
    )


def _resolve_tracking_backend(runtime_environment: MlflowRuntimeEnvironment) -> str:
    """Resolve the dlkit tracking backend, guarding against an implicit local fallback.

    Reads the backend from ``configs/tracking.toml`` so it comes from config, not
    code. When MLflow tracking is enabled, refuses to proceed if *runtime_environment*
    resolved to the local sqlite fallback rather than an explicit source
    (``MLFLOW_TRACKING_URI`` env var or ``configs/tracking.toml``'s ``uri``) —
    silently writing runs to local storage is worse than failing loudly.

    Args:
        runtime_environment: The resolved MLflow runtime environment for this run.

    Returns:
        The dlkit tracking backend identifier (e.g. ``"mlflow"`` or ``"none"``).

    Raises:
        RuntimeError: If MLflow tracking is enabled but no explicit tracking URI
            was ever configured.
    """
    shared_tracking = load_tracking_config()
    backend = shared_tracking.backend if shared_tracking else "mlflow"
    if backend == "mlflow" and not runtime_environment.is_explicit:
        raise RuntimeError(
            "MLflow tracking is enabled but no explicit tracking URI was "
            "configured (checked MLFLOW_TRACKING_URI env var and "
            "configs/tracking.toml's [tracking] uri). Refusing to silently "
            f"fall back to local storage at '{runtime_environment.tracking_uri}' — set "
            "the URI explicitly if this is intentional."
        )
    return backend


@dataclass(frozen=True)
class _TrainingFinalizationContext:
    """Everything ``_finalize_training_run`` needs from ``train_model``'s scope.

    Bundles the post-``execute()`` state into one value object so the
    "make a completed execution durable in MLflow" responsibility doesn't have to
    read closure locals off ``train_model``.

    Attributes:
        training_result: DLKit training result object returned by ``execute()``.
        run_config: Declarative MLflow run settings built before ``execute()`` ran.
        workspace: Filesystem workspace for this assignment's training run.
        assignment: Fully resolved assignment (spec/workspace/settings) this
            training run belongs to.
        assignment_id: The outer ``train_model`` call's own ``assignment_id``
            parameter. Used only in the final error message; may differ from
            ``assignment.spec.assignment_id`` since this can be ``None`` while
            ``load_assignment`` resolves a synthesized id.
        resolved_assignment_display_name: Human-readable assignment name.
        dataset_id: Dataset identity resolved from the workspace.
        resolved_dataset_display_name: Human-readable dataset name.
        workflow_settings: DLKit workflow settings used for this training run.
        contract: Runtime dataset contract for array/key resolution.
        config_path: Path to the job TOML config.
        resolved_data_config_path: Path to the dataset TOML config.
        fallback_tracking_uri: Tracking URI resolved from the runtime environment
            before ``execute()``, used as the starting point before
            ``_resolve_mlflow_run_ids`` may override it.
    """

    training_result: Any
    run_config: MlflowRunConfig
    workspace: AssignmentWorkspace
    assignment: RunnableAssignment
    assignment_id: str | None
    resolved_assignment_display_name: str
    dataset_id: str
    resolved_dataset_display_name: str
    workflow_settings: AnyJobConfig
    contract: RuntimeDatasetContract
    config_path: Path
    resolved_data_config_path: Path | None
    fallback_tracking_uri: str | None


@dataclass(frozen=True)
class PreparedTraining:
    """Fully-resolved training inputs, ready for a single ``execute()`` call.

    Owns a temporary workspace directory (``tmp_path``) that must outlive both
    the ``execute()`` call and the subsequent ``finalize_prepared_training()``
    call — release it with ``cleanup_prepared_training()`` once finalization
    has completed (success or failure).
    """

    workflow_settings: AnyJobConfig
    run_config: MlflowRunConfig
    runtime_mlflow_env: MlflowRuntimeEnvironment
    tmp_path: Path
    workspace: AssignmentWorkspace
    assignment: RunnableAssignment
    resolved_assignment_display_name: str
    dataset_id: str
    resolved_dataset_display_name: str
    contract: RuntimeDatasetContract
    config_path: Path
    resolved_data_config_path: Path


def prepare_training_settings(
    *,
    config_path: str | Path,
    data_config_path: str | Path,
    settings: NeurallsSettings | None = None,
    case_config_path: str | Path | None = None,
    output_root: Path | str | None = None,
    max_epochs: int | None = None,
    assignment_id: str | None = None,
    assignment_display_name: str | None = None,
    dataset_registry_id: str | None = None,
    dataset_display_name: str | None = None,
    job_registry_id: str | None = None,
    job_display_name: str | None = None,
    mlflow_experiment_name: str | None = None,
    batched: bool = False,
    extra_tags: Mapping[str, str] | None = None,
) -> PreparedTraining:
    """Resolve one assignment's dataset, config, and dlkit settings for training.

    This is steps 1-4 of what ``train_model()`` used to do inline: load the
    assignment config, resolve/stage dataset artifacts, build the execute()-time
    MLflow naming/tags, and configure dlkit's settings. Splitting it out lets a
    batch caller build every assignment's settings up front and hand them to
    dlkit's ``run_multirun_spec()`` as a single sweep, instead of calling
    ``execute()`` once per assignment itself.

    The returned ``PreparedTraining.tmp_path`` is a real (not context-managed)
    temp directory — the caller must eventually call
    ``cleanup_prepared_training()`` on the result, after ``execute()`` and
    ``finalize_prepared_training()`` have both run.

    Args:
        config_path: Path to a job configuration TOML.
        data_config_path: Path to a dataset configuration TOML.
        batched: True when this assignment is one child of a batch/sweep —
            omits the run-name timestamp, since the sweep already
            disambiguates children without one.
        extra_tags: Additional MLflow tags merged onto the run's tag set
            (e.g. a dataset content fingerprint for reuse-check invalidation).

    Returns:
        PreparedTraining ready to pass to ``execute()`` and then
        ``finalize_prepared_training()``.
    """
    resolved_case_config_path = Path(case_config_path) if case_config_path else None
    settings = require_settings(settings, case_config_path=resolved_case_config_path)
    runtime_environment = build_runtime_environment(
        output_root,
        default_output_root=settings.output_dir,
    )
    runtime_mlflow_env = runtime_environment.env

    tmp_path = Path(tempfile.mkdtemp(prefix="neuralls_train_"))
    try:
        config_path = Path(config_path)
        resolved_data_config_path = Path(data_config_path)
        # Step 1: Load assignment configuration into temp dir
        assignment = load_assignment(
            job_config_path=config_path,
            data_config_path=resolved_data_config_path,
            neuralls_settings=settings,
            case_config_path=resolved_case_config_path,
            output_root=tmp_path,
            assignment_id=assignment_id,
            assignment_display_name=assignment_display_name,
            dataset_registry_id=dataset_registry_id,
            dataset_display_name=dataset_display_name,
            job_registry_id=job_registry_id,
            job_display_name=job_display_name,
        )
        workflow_settings = assignment.settings
        workspace = assignment.workspace
        contract = default_training_dataset_contract()
        dataset_id = workspace.dataset_id
        resolved_assignment_display_name = assignment.spec.assignment_display_name
        resolved_dataset_display_name = assignment.spec.dataset_display_name or dataset_id

        # Step 2: Resolve training dataset artifacts
        arrays, features, targets = _load_and_prepare_data(workflow_settings, workspace, contract)
        dataset_format = resolve_dataset_format(arrays)

        # Step 3: Build execute()-time MLflow naming and tags
        run_config = _build_training_run_config(
            assignment_id=assignment.spec.assignment_id,
            assignment_display_name=resolved_assignment_display_name,
            dataset_registry_id=assignment.spec.dataset_id,
            job_registry_id=assignment.spec.job_id,
            dataset_display_name=resolved_dataset_display_name,
            mlflow_experiment_name=mlflow_experiment_name,
            runtime_mlflow_env=runtime_mlflow_env,
            workspace_root=workspace.root_dir,
            include_timestamp=not batched,
        )
        if extra_tags:
            run_config = replace(run_config, tags={**run_config.tags, **extra_tags})

        # Step 4: Configure DLKit settings (dataset, paths, MLflow names)
        workflow_settings, workspace = _configure_training_pipeline(
            workflow_settings,
            workspace,
            features,
            targets,
            contract,
            dataset_format,
        )

        if max_epochs is not None:
            workflow_settings = patch_model(
                workflow_settings,
                {"training": {"trainer": {"max_epochs": max_epochs}}},
            )
        tracking_backend = _resolve_tracking_backend(runtime_environment)
        workflow_settings = patch_model(
            workflow_settings,
            {
                "tracking": {
                    "backend": tracking_backend,
                    "uri": runtime_environment.tracking_uri,
                }
            },
        )
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    return PreparedTraining(
        workflow_settings=workflow_settings,
        run_config=run_config,
        runtime_mlflow_env=runtime_environment,
        tmp_path=tmp_path,
        workspace=workspace,
        assignment=assignment,
        resolved_assignment_display_name=resolved_assignment_display_name,
        dataset_id=dataset_id,
        resolved_dataset_display_name=resolved_dataset_display_name,
        contract=contract,
        config_path=config_path,
        resolved_data_config_path=resolved_data_config_path,
    )


def cleanup_prepared_training(prepared: PreparedTraining) -> None:
    """Remove a prepared training run's temporary workspace directory.

    Call once ``finalize_prepared_training()`` (or an earlier failure) has
    finished needing ``prepared.tmp_path`` on disk.
    """
    shutil.rmtree(prepared.tmp_path, ignore_errors=True)


def to_run_spec(prepared: PreparedTraining) -> RunSpec:
    """Build a dlkit multirun ``RunSpec`` for one already-prepared assignment.

    Reuses the same naming/tags ``execute()`` would otherwise receive via
    ``ExecutionOverrides`` — dlkit's ``MultiRunOrchestrator`` merges
    ``RunSpec.tags`` into the child's own ``experiment.tags`` and applies
    ``run_name``/experiment identically to what a single ``execute()`` call
    does today.
    """
    return RunSpec(
        id=prepared.assignment.spec.assignment_id,
        label=prepared.resolved_assignment_display_name,
        settings=prepared.workflow_settings,
        run_name=prepared.run_config.run_name,
        tags=dict(prepared.run_config.tags),
    )


def finalize_prepared_training(
    prepared: PreparedTraining,
    execution_result: object,
) -> tuple[str, str]:
    """Finalize one dlkit ``execute()`` result: make the checkpoint durable in MLflow.

    Must be called while ``prepared.tmp_path`` still exists on disk — call
    ``cleanup_prepared_training(prepared)`` afterward regardless of outcome.

    Args:
        prepared: The assignment's prepared training inputs.
        execution_result: Whatever dlkit's ``execute()`` (or a multirun
            child's dispatch) returned for this assignment.

    Returns:
        Tuple of (run_id, tracking_uri) for the MLflow run the checkpoint was
        uploaded to.
    """
    training_result = _unwrap_execution_result(execution_result)
    fallback_tracking_uri, _ = resolve_runtime_tracking_config()
    context = _TrainingFinalizationContext(
        training_result=training_result,
        run_config=prepared.run_config,
        workspace=prepared.workspace,
        assignment=prepared.assignment,
        assignment_id=prepared.assignment.spec.assignment_id,
        resolved_assignment_display_name=prepared.resolved_assignment_display_name,
        dataset_id=prepared.dataset_id,
        resolved_dataset_display_name=prepared.resolved_dataset_display_name,
        workflow_settings=prepared.workflow_settings,
        contract=prepared.contract,
        config_path=prepared.config_path,
        resolved_data_config_path=prepared.resolved_data_config_path,
        fallback_tracking_uri=fallback_tracking_uri,
    )
    return _finalize_training_run(context)


def _finalize_training_run(context: _TrainingFinalizationContext) -> tuple[str, str]:
    """Make a completed execution durable in MLflow, or mark the run FAILED and re-raise.

    Must be called from inside the same ``scoped_mlflow_environment`` the
    execution ran under. dlkit closes the MLflow run's status to ``FINISHED``
    synchronously as soon as ``execute()`` returns, before any of this function's
    work (resolving run metadata, uploading the checkpoint, staging other
    artifacts) has happened. If any of that work raises after a run_id has been
    resolved, this marks that run ``FAILED`` (best-effort, via ``mark_run_failed``)
    before re-raising the original exception unchanged — a broken durability step
    must never leave a run reading as ``FINISHED``-and-usable. If no run_id was
    ever resolved, nothing is marked (there is nothing to attach the failure to).

    Args:
        context: Bundled state produced by ``train_model`` after ``execute()``.

    Returns:
        Tuple of (run_id, tracking_uri) for the MLflow run the checkpoint was
        uploaded to.

    Raises:
        RuntimeError: If no MLflow run could be established for the assignment.
    """
    resolved_run_id: str | None = None
    resolved_tracking_uri: str | None = context.fallback_tracking_uri
    try:
        # Step 6: Resolve MLflow run metadata and retrieve the checkpoint
        mlflow_coords = _resolve_mlflow_run_ids(
            training_result=context.training_result,
            fallback_tracking_uri=context.fallback_tracking_uri,
            experiment_name=context.run_config.experiment_name,
            run_name=context.run_config.run_name,
        )
        if mlflow_coords is not None:
            resolved_tracking_uri, _, resolved_run_id = mlflow_coords
        else:
            resolved_run_id = getattr(context.training_result, "mlflow_run_id", None) or getattr(
                context.training_result, "run_id", None
            )
        with _training_artifact_lease_scope(
            tracking_uri=resolved_tracking_uri,
            run_id=resolved_run_id,
        ) as artifact_leases:
            local_checkpoint = _resolve_finalization_checkpoint(
                context=context,
                run_id=resolved_run_id,
                artifact_leases=artifact_leases,
            )
            match mlflow_coords:
                case (str(), str(), str()) as existing_coords:
                    mlflow_coords = _finalize_existing_mlflow_run(
                        context=context,
                        mlflow_coords=existing_coords,
                        checkpoint_path=local_checkpoint,
                    )
                case None:
                    mlflow_coords = _finalize_fallback_mlflow_run(
                        context=context,
                        checkpoint_path=local_checkpoint,
                    )
                case _:
                    raise RuntimeError(f"Invalid MLflow coordinates: {mlflow_coords!r}")

        if mlflow_coords is None:
            raise RuntimeError(
                f"Training completed but no MLflow run could be established for "
                f"assignment '{context.assignment_id}' — nothing to attach the checkpoint to."
            )
        resolved_final_tracking_uri, _, resolved_final_run_id = mlflow_coords
        return resolved_final_run_id, resolved_final_tracking_uri
    except Exception:
        if resolved_run_id is not None and resolved_tracking_uri is not None:
            mark_run_failed(run_id=resolved_run_id, tracking_uri=resolved_tracking_uri)
        raise
