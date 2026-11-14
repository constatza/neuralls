"""Eval-only preparation: turn a case and its assignments into dlkit run specs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dlkit.engine.workflows.multi_run import RunSpec
from dlkit.infrastructure.config.data_entries import DataEntry
from dlkit.infrastructure.config.job_config import InferenceJobConfig
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments._dataset_assembly import (
    _load_and_prepare_data,
    resolve_dataset_format,
)
from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments._registry_lookup import _resolve_config_paths
from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment
from neuralls.composition.assignments.job_loader import load_experiment_job
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.composition.assignments.runtime_dataset_patcher import patch_runtime_dataset
from neuralls.composition.assignments.runtime_workspace_patcher import patch_dataloader_runtime
from neuralls.composition.identity.training import training_identity
from neuralls.composition.tracking.run_specs import (
    build_evaluation_run,
)
from neuralls.composition.tracking.services import MLFLOW_TRACKING, TrackingServices
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.run_names import TrainingPresentation
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
)
from neuralls.platform.config.resolution import (
    derive_output_root_from_tracking_uri,
    is_sqlite_tracking_uri,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.tracking.artifact_access import ArtifactLeaseManager
from neuralls.platform.tracking.evaluation_artifacts import (
    TrainingEvaluationArtifacts,
    resolve_training_evaluation_artifacts,
)
from neuralls.platform.tracking.run_names import render_training_label, run_name_text

DEFAULT_EVAL_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class EvaluationConfigPaths:
    """Resolved case and staged config paths for one eval assignment."""

    job_config_path: Path
    data_config_path: Path
    staged_job_config_path: Path
    staged_data_config_path: Path


@dataclass(frozen=True, slots=True)
class EvaluationAssignmentContext:
    """Runtime context needed to evaluate one assignment."""

    client: MlflowClient
    training_run_id: str
    artifacts: TrainingEvaluationArtifacts
    config_paths: EvaluationConfigPaths


@dataclass(frozen=True, slots=True)
class PreparedEvaluation:
    """One assignment's fully-resolved eval settings, ready for a dlkit RunSpec."""

    inference_settings: InferenceJobConfig
    assignment: AssignmentEntry
    presentation: TrainingPresentation
    training_run_id: str
    client: MlflowClient
    split_file: Path
    split_artifact_path: str
    output_root: Path


@dataclass(frozen=True, slots=True)
class EvaluationPreparationBatch:
    """Prepared eval children plus whether any assignment failed during preparation."""

    prepared_by_child_id: dict[str, PreparedEvaluation]
    run_specs: list[RunSpec]
    any_failed: bool


def _with_eval_runtime_dataset(
    settings: AnyJobConfig,
    *,
    split_file: Path,
    checkpoint_path: Path,
    features: list[DataEntry],
    targets: list[DataEntry],
    dataset_format: str,
) -> AnyJobConfig:
    contract = default_training_dataset_contract()
    settings = patch_runtime_dataset(
        settings,
        features=features,
        targets=targets,
        contract=contract,
    )
    settings = patch_dataloader_runtime(settings, dataset_format=dataset_format)
    return settings.patch(
        {
            "run": {"type": "predict"},
            "model": {"checkpoint": str(checkpoint_path)},
            "data": {
                "splits": {"filepath": split_file},
                "module": {
                    "name": "ArrayDataModule",
                    "module_path": "dlkit.engine.adapters.lightning.datamodules",
                },
            },
        }
    )


def _resolve_training_run_id(
    *,
    tracking: TrackingServices,
    tracking_uri: str,
    experiment_name: str,
    assignment_id: str,
    identity: StageIdentity,
) -> str:
    """Find the training run carrying this assignment's derived identity.

    Args:
        tracking: Source of the training-run lookup.
        tracking_uri: MLflow tracking URI.
        experiment_name: Training experiment.
        assignment_id: Assignment label, for the error message only.
        identity: The assignment's training identity (dataset content,
            effective job settings).

    Returns:
        The producing run's id.

    Raises:
        RuntimeError: If no FINISHED run with a checkpoint carries the identity.
    """
    reused = tracking.identity_store(
        tracking_uri=tracking_uri, experiment=experiment_name, require_checkpoint=True
    ).find(identity)
    if reused is None or reused.run_id is None:
        raise RuntimeError(
            f"No FINISHED training run matching the current dataset and job settings found for "
            f"assignment '{assignment_id}' in MLflow experiment '{experiment_name}'."
        )
    return reused.run_id


def _resolve_eval_config_paths(
    *,
    assignment: AssignmentEntry,
    configs_dir: Path,
    cfg: CaseConfig,
    artifacts: TrainingEvaluationArtifacts,
    training_run_id: str,
) -> EvaluationConfigPaths:
    job_config_path, data_config_path = _resolve_config_paths(assignment, configs_dir, cfg)
    staged_job_config_path = artifacts.config_artifacts.find_named(job_config_path)
    staged_data_config_path = artifacts.config_artifacts.find_named(data_config_path)
    if staged_job_config_path == job_config_path:
        logger.warning(
            "Training run '{}' has no staged job config named '{}'; using case config path.",
            training_run_id,
            job_config_path.name,
        )
    if staged_data_config_path == data_config_path:
        logger.warning(
            "Training run '{}' has no staged dataset config named '{}'; using case config path.",
            training_run_id,
            data_config_path.name,
        )
    return EvaluationConfigPaths(
        job_config_path=job_config_path,
        data_config_path=data_config_path,
        staged_job_config_path=staged_job_config_path,
        staged_data_config_path=staged_data_config_path,
    )


def _build_eval_context(
    *,
    cfg: CaseConfig,
    configs_dir: Path,
    assignment: AssignmentEntry,
    tracking_uri: str,
    tracking: TrackingServices,
    artifact_leases: ArtifactLeaseManager,
    settings: NeurallsSettings,
) -> EvaluationAssignmentContext:
    job_config_path, data_config_path = _resolve_config_paths(assignment, configs_dir, cfg)
    training_run_id = _resolve_training_run_id(
        tracking=tracking,
        tracking_uri=tracking_uri,
        experiment_name=cfg.names.training,
        assignment_id=assignment.id,
        identity=training_identity(
            job_config_path=job_config_path,
            data_config_path=data_config_path,
            settings=settings,
        ),
    )
    client = tracking.client(tracking_uri)
    artifacts = resolve_training_evaluation_artifacts(
        client=client,
        run_id=training_run_id,
        artifact_leases=artifact_leases,
        assignment_id=assignment.id,
    )
    config_paths = _resolve_eval_config_paths(
        assignment=assignment,
        configs_dir=configs_dir,
        cfg=cfg,
        artifacts=artifacts,
        training_run_id=training_run_id,
    )
    return EvaluationAssignmentContext(
        client=client,
        training_run_id=training_run_id,
        artifacts=artifacts,
        config_paths=config_paths,
    )


def _materialize_inference_settings(
    *,
    cfg: CaseConfig,
    settings: NeurallsSettings,
    assignment: AssignmentEntry,
    output_root: Path,
    case_config_path: Path,
    context: EvaluationAssignmentContext,
) -> InferenceJobConfig:
    runnable = load_assignment(
        job_config_path=context.config_paths.staged_job_config_path,
        data_config_path=context.config_paths.staged_data_config_path,
        neuralls_settings=settings,
        output_root=output_root,
        mode="inference",
        case_config_path=case_config_path,
        identity=AssignmentIdentity(
            assignment_id=assignment.id,
            dataset_registry_id=assignment.dataset_id,
            job_registry_id=assignment.job_id,
        ),
    )
    job_settings = load_experiment_job(context.config_paths.staged_job_config_path, settings)
    contract = default_training_dataset_contract()
    arrays, features, targets = _load_and_prepare_data(job_settings, runnable.workspace, contract)
    eval_settings = _with_eval_runtime_dataset(
        job_settings,
        split_file=context.artifacts.split_file,
        checkpoint_path=context.artifacts.checkpoint_path,
        features=features,
        targets=targets,
        dataset_format=resolve_dataset_format(arrays),
    )
    eval_settings = eval_settings.patch({"experiment": {"name": cfg.names.training}})
    return _as_inference_job(eval_settings)


def _as_inference_job(settings: AnyJobConfig) -> InferenceJobConfig:
    return InferenceJobConfig(
        run=settings.run,
        experiment=settings.experiment,
        model=settings.model,
        data=settings.data,
        training=getattr(settings, "training", None),
        search=getattr(settings, "search", None),
        tracking=settings.tracking,
        plots=settings.plots,
    )


def prepare_evaluation_settings(
    *,
    cfg: CaseConfig,
    configs_dir: Path,
    settings: NeurallsSettings,
    assignment: AssignmentEntry,
    presentation: TrainingPresentation,
    output_root: Path,
    case_config_path: Path,
    tracking_uri: str,
    artifact_leases: ArtifactLeaseManager,
    tracking: TrackingServices = MLFLOW_TRACKING,
) -> PreparedEvaluation:
    """Resolve one assignment's checkpoint, split, and dlkit eval settings.

    Building a distinct InferenceJobConfig per assignment (vs. one settings
    object shared across the sweep) is what lets each child evaluate against
    the dataset it was actually trained on.
    """
    context = _build_eval_context(
        cfg=cfg,
        configs_dir=configs_dir,
        assignment=assignment,
        tracking_uri=tracking_uri,
        tracking=tracking,
        artifact_leases=artifact_leases,
        settings=settings,
    )
    inference_settings = _materialize_inference_settings(
        cfg=cfg,
        settings=settings,
        assignment=assignment,
        output_root=output_root,
        case_config_path=case_config_path,
        context=context,
    )
    data_settings = inference_settings.data
    batch_size = (
        data_settings.batch_size if data_settings is not None else None
    ) or DEFAULT_EVAL_BATCH_SIZE
    inference_settings = inference_settings.patch(
        {"tracking": {"backend": "mlflow"}, "data": {"batch_size": batch_size}}
    )
    return PreparedEvaluation(
        inference_settings=inference_settings,
        assignment=assignment,
        presentation=presentation,
        training_run_id=context.training_run_id,
        client=context.client,
        split_file=context.artifacts.split_file,
        split_artifact_path=context.artifacts.split_artifact_path,
        output_root=output_root,
    )


def to_eval_run_spec(prepared: PreparedEvaluation) -> RunSpec:
    """Build a dlkit multirun ``RunSpec`` for one already-prepared evaluation."""
    return RunSpec(
        id=prepared.assignment.id,
        label=render_training_label(prepared.presentation),
        settings=prepared.inference_settings,
        run_name=run_name_text(build_evaluation_run(prepared.presentation)),
    )


def resolve_base_output(
    cfg: CaseConfig,
    settings: NeurallsSettings,
    output_root: Path | None,
    configs_dir: Path,
) -> Path:
    if output_root is not None:
        return output_root
    tracking_uri = cfg.mlflow.tracking_uri
    if tracking_uri is not None and is_sqlite_tracking_uri(tracking_uri):
        derived = derive_output_root_from_tracking_uri(tracking_uri, base_dir=configs_dir)
        if derived is None:
            raise ValueError("MLflow tracking_uri must resolve to a local sqlite database path.")
        return derived
    return settings.output_dir


def prepare_eval_run_specs(
    *,
    assignments: list[AssignmentEntry],
    presentations: Mapping[str, TrainingPresentation],
    cfg: CaseConfig,
    configs_dir: Path,
    settings: NeurallsSettings,
    output_root: Path,
    case_config_path: Path,
    tracking: TrackingServices,
    tracking_uri: str,
    artifact_leases: ArtifactLeaseManager,
) -> EvaluationPreparationBatch:
    prepared_by_child_id: dict[str, PreparedEvaluation] = {}
    run_specs: list[RunSpec] = []
    any_failed = False
    for assignment in assignments:
        try:
            prepared = prepare_evaluation_settings(
                cfg=cfg,
                configs_dir=configs_dir,
                settings=settings,
                assignment=assignment,
                presentation=presentations[assignment.id],
                output_root=output_root,
                case_config_path=case_config_path,
                tracking=tracking,
                tracking_uri=tracking_uri,
                artifact_leases=artifact_leases,
            )
        except Exception as exc:  # noqa: BLE001
            any_failed = True
            logger.error("{} failed: {}", render_training_label(presentations[assignment.id]), exc)
            continue
        prepared_by_child_id[assignment.id] = prepared
        run_specs.append(to_eval_run_spec(prepared))
    return EvaluationPreparationBatch(
        prepared_by_child_id=prepared_by_child_id,
        run_specs=run_specs,
        any_failed=any_failed,
    )
