"""Eval-only assignment workflow backed by DLKit checkpoint evaluation."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dlkit.common import ChildFailure, ChildSuccess
from dlkit.common.results import EvaluationResult
from dlkit.engine.workflows.multi_run import MultiRunSpec, RunSpec
from dlkit.infrastructure.config.data_entries import DataEntry
from dlkit.infrastructure.config.job_config import InferenceJobConfig
from dlkit.interfaces.api import run_multirun_spec
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments._dataset_assembly import (
    _load_and_prepare_data,
    resolve_dataset_format,
)
from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments._registry_lookup import (
    _find_registry_entry,
    _resolve_config_paths,
)
from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment
from neuralls.composition.assignments.job_loader import load_experiment_job
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.composition.assignments.runtime_dataset_patcher import patch_runtime_dataset
from neuralls.composition.assignments.runtime_workspace_patcher import patch_dataloader_runtime
from neuralls.composition.tracking.run_specs import (
    build_evaluation_tags,
    build_session_run_spec,
)
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
)
from neuralls.platform.config.resolution import (
    derive_output_root_from_tracking_uri,
    is_sqlite_tracking_uri,
)
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.plots import plot_metric_comparison
from neuralls.platform.tracking.artifact_access import (
    ArtifactLeaseManager,
    MlflowArtifactLeaseManager,
)
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.evaluation_artifacts import (
    TrainingEvaluationArtifacts,
    resolve_training_evaluation_artifacts,
)
from neuralls.platform.tracking.mlflow import (
    build_workflow_environment,
    create_session_parent_run,
    finalize_session_parent_run,
)
from neuralls.platform.tracking.mlflow_client import (
    find_successful_run,
    log_batch_artifacts_to_mlflow,
)

type MlflowClientFactory = Callable[..., MlflowClient]

DEFAULT_EVAL_BATCH_SIZE = 32
EVAL_OUTPUT_DIR_NAME = "eval"


@dataclass(frozen=True)
class EvaluationRunResult:
    """Immutable result for one eval-only assignment run."""

    label: str
    assignment_id: str
    assignment_display_name: str
    training_run_id: str
    evaluation_run_id: str | None
    metrics: dict[str, float]
    split_file: Path
    split_artifact_path: str
    figures_dir: Path


@dataclass(frozen=True)
class EvaluationBatchResult:
    """Immutable result for a completed eval-only batch."""

    results: list[EvaluationRunResult]
    label_map: dict[str, dict[str, str | None]]
    output_dir: Path
    tracking_uri: str
    parent_run_id: str


@dataclass(frozen=True)
class EvaluationConfigPaths:
    """Resolved case and staged config paths for one eval assignment."""

    job_config_path: Path
    data_config_path: Path
    staged_job_config_path: Path
    staged_data_config_path: Path


@dataclass(frozen=True)
class EvaluationAssignmentContext:
    """Runtime context needed to evaluate one assignment."""

    client: MlflowClient
    training_run_id: str
    artifacts: TrainingEvaluationArtifacts
    config_paths: EvaluationConfigPaths
    dataset_display_name: str | None
    job_display_name: str | None


@dataclass(frozen=True)
class PreparedEvaluation:
    """One assignment's fully-resolved eval settings, ready for a dlkit RunSpec."""

    inference_settings: InferenceJobConfig
    assignment: AssignmentEntry
    resolved_assignment_display_name: str
    training_run_id: str
    client: MlflowClient
    split_file: Path
    split_artifact_path: str
    output_root: Path
    label: str
    run_name: str


@dataclass(frozen=True)
class EvaluationPreparationBatch:
    """Prepared eval children plus whether any assignment failed during preparation."""

    prepared_by_child_id: dict[str, PreparedEvaluation]
    run_specs: list[RunSpec]
    any_failed: bool


def _sanitize_path_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "assignment"


def _resolve_display_metadata(
    entry: AssignmentEntry,
    cfg: CaseConfig,
) -> tuple[str | None, str | None]:
    dataset_entry = _find_registry_entry(cfg.datasets, entry.dataset_id)
    job_entry = _find_registry_entry(cfg.jobs, entry.job_id)
    dataset_display = (
        dataset_entry.effective_display_name if dataset_entry is not None else entry.dataset_id
    )
    job_display = job_entry.effective_display_name if job_entry is not None else entry.job_id
    return dataset_display, job_display


def _make_label_map(
    results: Sequence[EvaluationRunResult],
) -> dict[str, dict[str, str | None]]:
    return {
        result.label: {
            "assignment_id": result.assignment_id,
            "assignment_display_name": result.assignment_display_name,
            "training_run_id": result.training_run_id,
            "evaluation_run_id": result.evaluation_run_id,
            "split_artifact_path": result.split_artifact_path,
        }
        for result in results
    }


def _metric_value(metrics: dict[str, float], metric: str) -> float | None:
    if metric in metrics:
        return metrics[metric]
    prefixed = f"eval/{metric}"
    return metrics.get(prefixed)


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
    tracking_uri: str,
    experiment_name: str,
    assignment_id: str,
) -> str:
    training_run_id = find_successful_run(
        tracking_uri=tracking_uri,
        mlflow_experiment_name=experiment_name,
        assignment_id=assignment_id,
    )
    if training_run_id is None:
        raise RuntimeError(
            f"No FINISHED training run found for assignment '{assignment_id}' "
            f"in MLflow experiment '{experiment_name}'."
        )
    return training_run_id


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
    mlflow_client_factory: MlflowClientFactory,
    artifact_leases: ArtifactLeaseManager,
) -> EvaluationAssignmentContext:
    training_run_id = _resolve_training_run_id(
        tracking_uri=tracking_uri,
        experiment_name=cfg.names.training,
        assignment_id=assignment.id,
    )
    client = mlflow_client_factory(tracking_uri=tracking_uri)
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
    dataset_display_name, job_display_name = _resolve_display_metadata(assignment, cfg)
    return EvaluationAssignmentContext(
        client=client,
        training_run_id=training_run_id,
        artifacts=artifacts,
        config_paths=config_paths,
        dataset_display_name=dataset_display_name,
        job_display_name=job_display_name,
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
            assignment_display_name=assignment.effective_display_name,
            dataset_registry_id=assignment.dataset_id,
            dataset_display_name=context.dataset_display_name,
            job_registry_id=assignment.job_id,
            job_display_name=context.job_display_name,
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


def _save_eval_outputs(
    result: EvaluationResult,
    *,
    assignment_dir: Path,
) -> Path:
    metrics_dir = assignment_dir / "metrics"
    figures_dir = assignment_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "evaluation_metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for key, figure in result.figures.items():
        savefig = getattr(figure, "savefig", None)
        if callable(savefig):
            savefig(figures_dir / f"{_sanitize_path_part(key)}.png")
    return figures_dir


def _tag_evaluation_run(
    *,
    client: MlflowClient,
    run_id: str | None,
    assignment_id: str,
    assignment_display_name: str,
    training_run_id: str,
    split_artifact_path: str,
) -> None:
    """Set bookkeeping tags/params dlkit has no concept of on a completed eval run."""
    if run_id is None:
        return
    tags = build_evaluation_tags(
        assignment_id=assignment_id,
        assignment_display_name=assignment_display_name,
        source_training_run_id=training_run_id,
        split_artifact=split_artifact_path,
    ).as_mlflow_tags()
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    client.log_param(run_id, "source_training_run_id", training_run_id)
    client.log_param(run_id, "split_artifact_path", split_artifact_path)


def prepare_evaluation_settings(
    *,
    cfg: CaseConfig,
    configs_dir: Path,
    settings: NeurallsSettings,
    assignment: AssignmentEntry,
    output_root: Path,
    case_config_path: Path,
    tracking_uri: str,
    label: str,
    artifact_leases: ArtifactLeaseManager,
    mlflow_client_factory: MlflowClientFactory = MlflowClient,
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
        mlflow_client_factory=mlflow_client_factory,
        artifact_leases=artifact_leases,
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
        resolved_assignment_display_name=assignment.effective_display_name,
        training_run_id=context.training_run_id,
        client=context.client,
        split_file=context.artifacts.split_file,
        split_artifact_path=context.artifacts.split_artifact_path,
        output_root=output_root,
        label=label,
        run_name=assignment.effective_display_name,
    )


def to_eval_run_spec(prepared: PreparedEvaluation) -> RunSpec:
    """Build a dlkit multirun ``RunSpec`` for one already-prepared evaluation."""
    return RunSpec(
        id=prepared.assignment.id,
        label=prepared.resolved_assignment_display_name,
        settings=prepared.inference_settings,
        run_name=prepared.run_name,
    )


def _finalize_eval_child(
    prepared: PreparedEvaluation,
    execution_result: object,
) -> EvaluationRunResult:
    """Finalize one successful multirun child: save outputs, tag, and build the result."""
    result = cast(EvaluationResult, execution_result)
    assignment_dir = (
        prepared.output_root / EVAL_OUTPUT_DIR_NAME / _sanitize_path_part(prepared.assignment.id)
    )
    figures_dir = _save_eval_outputs(result, assignment_dir=assignment_dir)
    _tag_evaluation_run(
        client=prepared.client,
        run_id=result.mlflow_run_id,
        assignment_id=prepared.assignment.id,
        assignment_display_name=prepared.resolved_assignment_display_name,
        training_run_id=prepared.training_run_id,
        split_artifact_path=prepared.split_artifact_path,
    )
    return EvaluationRunResult(
        label=prepared.label,
        assignment_id=prepared.assignment.id,
        assignment_display_name=prepared.resolved_assignment_display_name,
        training_run_id=prepared.training_run_id,
        evaluation_run_id=result.mlflow_run_id,
        metrics=dict(result.metrics),
        split_file=prepared.split_file,
        split_artifact_path=prepared.split_artifact_path,
        figures_dir=figures_dir,
    )


def _resolve_base_output(
    cfg: CaseConfig,
    settings: NeurallsSettings,
    output_root: Path | None,
) -> Path:
    if output_root is not None:
        return output_root
    tracking_uri = cfg.mlflow.tracking_uri
    if tracking_uri is not None and is_sqlite_tracking_uri(tracking_uri):
        derived = derive_output_root_from_tracking_uri(tracking_uri)
        if derived is None:
            raise ValueError("MLflow tracking_uri must resolve to a local sqlite database path.")
        return derived
    return settings.output_dir


def _prepare_eval_run_specs(
    *,
    assignments: list[AssignmentEntry],
    cfg: CaseConfig,
    configs_dir: Path,
    settings: NeurallsSettings,
    output_root: Path,
    case_config_path: Path,
    tracking_uri: str,
    artifact_leases: ArtifactLeaseManager,
) -> EvaluationPreparationBatch:
    prepared_by_child_id: dict[str, PreparedEvaluation] = {}
    run_specs: list[RunSpec] = []
    any_failed = False
    for index, assignment in enumerate(assignments, start=1):
        try:
            prepared = prepare_evaluation_settings(
                cfg=cfg,
                configs_dir=configs_dir,
                settings=settings,
                assignment=assignment,
                output_root=output_root,
                case_config_path=case_config_path,
                tracking_uri=tracking_uri,
                label=str(index),
                artifact_leases=artifact_leases,
            )
        except Exception as exc:  # noqa: BLE001
            any_failed = True
            logger.error("Eval assignment '{}' failed: {}", assignment.id, exc)
            continue
        prepared_by_child_id[assignment.id] = prepared
        run_specs.append(to_eval_run_spec(prepared))
    return EvaluationPreparationBatch(
        prepared_by_child_id=prepared_by_child_id,
        run_specs=run_specs,
        any_failed=any_failed,
    )


def _run_prepared_eval_sweep(
    *,
    preparation: EvaluationPreparationBatch,
    experiment_name: str,
    parent_run_name: str,
    parent_tags: dict[str, str],
) -> tuple[Any | None, list[EvaluationRunResult], bool]:
    if not preparation.run_specs:
        return None, [], preparation.any_failed
    sweep_result = run_multirun_spec(
        MultiRunSpec(
            experiment_name=experiment_name,
            parent_run_name=parent_run_name,
            parent_tags=parent_tags,
            failure_policy="continue",
            children=tuple(preparation.run_specs),
        )
    )
    results: list[EvaluationRunResult] = []
    any_failed = preparation.any_failed
    for child_outcome in sweep_result.children:
        prepared = preparation.prepared_by_child_id[child_outcome.child_id]
        match child_outcome:
            case ChildSuccess():
                results.append(_finalize_eval_child(prepared, child_outcome.result))
            case ChildFailure():
                any_failed = True
                logger.error(
                    "Eval assignment '{}' failed: {}",
                    child_outcome.child_id,
                    child_outcome.message,
                )
    return sweep_result, results, any_failed


def _resolve_eval_parent_run(
    *,
    sweep_result: Any | None,
    tracking_uri: str,
    artifact_uri: str | None,
    run_name: str,
    tags: dict[str, str],
    experiment_name: str,
) -> tuple[str, str]:
    match sweep_result:
        case None:
            parent_run_id = create_session_parent_run(
                tracking_uri=tracking_uri,
                artifact_uri=artifact_uri,
                run_name=run_name,
                tags=tags,
                experiment_name=experiment_name,
            )
            return parent_run_id, tracking_uri
        case _:
            resolved_tracking_uri = sweep_result.tracking_uri or tracking_uri
            return sweep_result.parent_run_id, resolved_tracking_uri


def eval_batch(
    cfg: CaseConfig,
    configs_dir: Path,
    settings: NeurallsSettings | None = None,
    output_root: Path | None = None,
    case_config_path: Path | None = None,
    assignment_ids: Sequence[str] | None = None,
) -> EvaluationBatchResult:
    """Evaluate selected assignments from one case config as one dlkit multirun sweep."""
    settings = require_settings(settings, case_config_path=case_config_path)
    selected_ids = set(assignment_ids or ())
    assignments = [
        assignment
        for assignment in cfg.assignments
        if not selected_ids or assignment.id in selected_ids
    ]
    missing_ids = sorted(selected_ids - {assignment.id for assignment in cfg.assignments})
    if missing_ids:
        raise ValueError(f"Unknown assignment ids for eval: {', '.join(missing_ids)}.")
    if not assignments:
        raise ValueError("No [[assignments]] entries found in case config.")

    base_output = _resolve_base_output(cfg, settings, output_root)
    eval_mlflow_env = build_workflow_environment(
        tracking_uri=cfg.mlflow.tracking_uri,
        artifact_location=cfg.mlflow.artifacts_destination,
        default_output_root=base_output,
    )
    resolved_case_config_path = (
        case_config_path.resolve() if case_config_path is not None else (configs_dir / "case.toml")
    )
    mlflow_experiment_name = cfg.names.training

    with scoped_mlflow_environment(eval_mlflow_env.env):
        session_run_name, session_tags = build_session_run_spec(
            case_config_path=resolved_case_config_path,
            experiment_name=mlflow_experiment_name,
            phase="session_evaluation",
        )
        session_tag_map = dict(session_tags.as_mlflow_tags())

        lease_client = MlflowClient(tracking_uri=eval_mlflow_env.tracking_uri)
        with MlflowArtifactLeaseManager(client=lease_client) as artifact_leases:
            preparation = _prepare_eval_run_specs(
                assignments=assignments,
                cfg=cfg,
                configs_dir=configs_dir,
                settings=settings,
                output_root=base_output,
                case_config_path=resolved_case_config_path,
                tracking_uri=eval_mlflow_env.tracking_uri,
                artifact_leases=artifact_leases,
            )
            sweep_result, results, any_failed = _run_prepared_eval_sweep(
                preparation=preparation,
                experiment_name=mlflow_experiment_name,
                parent_run_name=session_run_name,
                parent_tags=session_tag_map,
            )

        parent_run_id, tracking_uri = _resolve_eval_parent_run(
            sweep_result=sweep_result,
            tracking_uri=eval_mlflow_env.tracking_uri,
            artifact_uri=eval_mlflow_env.artifact_uri,
            run_name=session_run_name,
            tags=session_tag_map,
            experiment_name=mlflow_experiment_name,
        )

        finalize_session_parent_run(
            tracking_uri=tracking_uri,
            run_id=parent_run_id,
            status="FAILED" if any_failed else "FINISHED",
        )

    return EvaluationBatchResult(
        results=results,
        label_map=_make_label_map(results),
        output_dir=base_output / EVAL_OUTPUT_DIR_NAME,
        tracking_uri=tracking_uri,
        parent_run_id=parent_run_id,
    )


def write_eval_metric_report(
    batch: EvaluationBatchResult,
    *,
    metric: str,
) -> bool:
    """Build eval-batch summary artifacts and log them to the batch's MLflow parent run.

    Stages the comparison barplot and label map in a scratch directory and
    uploads them via ``log_batch_artifacts_to_mlflow`` — nothing is left
    on local disk, matching the batch's other artifacts, which all live under
    the session parent run rather than a non-MLflow output directory.

    Args:
        batch: Completed eval batch result.
        metric: Metric key to plot across assignments.

    Returns:
        True if a comparison plot was produced (metric present for >=1 result).
    """
    labels: list[str] = []
    values: list[float] = []
    for result in batch.results:
        value = _metric_value(result.metrics, metric)
        if value is None:
            logger.warning(
                "Metric '{}' missing for eval assignment '{}'.",
                metric,
                result.assignment_display_name,
            )
            continue
        labels.append(result.label)
        values.append(value)

    plot_name = f"batch_metric_{metric.replace('/', '_')}.png"
    with tempfile.TemporaryDirectory() as tmp:
        work_root = Path(tmp)
        plotted = bool(labels)
        if plotted:
            legend = {
                result.label: (
                    f"{result.assignment_display_name} (run: {result.evaluation_run_id})"
                    if result.evaluation_run_id
                    else result.assignment_display_name
                )
                for result in batch.results
                if result.label in labels
            }
            plot_metric_comparison(
                labels=labels,
                values=values,
                metric_name=metric,
                legend=legend,
                save_path=work_root / plot_name,
            )

        (work_root / "batch_eval_labels.json").write_text(
            json.dumps(batch.label_map, indent=2),
            encoding="utf-8",
        )

        log_batch_artifacts_to_mlflow(
            tracking_uri=batch.tracking_uri,
            run_id=batch.parent_run_id,
            work_root=work_root,
            flat_files=(plot_name, "batch_eval_labels.json"),
        )

    return plotted
