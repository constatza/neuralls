"""Eval-only assignment workflow backed by DLKit checkpoint evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dlkit.common import ChildFailure, ChildSuccess
from dlkit.common.results import EvaluationResult
from dlkit.engine.workflows.multi_run import MultiRunSpec
from dlkit.interfaces.api import run_multirun_spec
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.assignment_subjects import (
    case_notation,
    training_presentations,
)
from neuralls.composition.assignments.evaluation_prepare import (
    EvaluationPreparationBatch,
    PreparedEvaluation,
    prepare_eval_run_specs,
    resolve_base_output,
)
from neuralls.composition.tracking.run_specs import (
    build_evaluation_tags,
    build_session_run_spec,
)
from neuralls.composition.tracking.services import MLFLOW_TRACKING, TrackingServices
from neuralls.domain.presentation.run_names import NamedRun, TrainingPresentation, training_dataset
from neuralls.domain.presentation.titles import ComparisonScopePresentation, session_title
from neuralls.platform.config.case_path import CaseConfigPath
from neuralls.platform.config.models.experiments import (
    CaseConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import (
    build_workflow_environment,
    create_session_parent_run,
    finalize_session_parent_run,
)
from neuralls.platform.tracking.run_names import render_training_label, run_name_text

EVAL_OUTPUT_DIR_NAME = "eval"


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """Immutable result for one eval-only assignment run."""

    assignment_id: str
    presentation: TrainingPresentation
    training_run_id: str
    evaluation_run_id: str | None
    metrics: dict[str, float]
    split_file: Path
    split_artifact_path: str
    figures_dir: Path


@dataclass(frozen=True, slots=True)
class EvaluationBatchResult:
    """Immutable result for a completed eval-only batch."""

    results: list[EvaluationRunResult]
    label_map: dict[str, dict[str, str | None]]
    output_dir: Path
    tracking_uri: str
    parent_run_id: str


def _sanitize_path_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "assignment"


def _make_label_map(
    results: Sequence[EvaluationRunResult],
) -> dict[str, dict[str, str | None]]:
    return {
        result.assignment_id: {
            "assignment_id": result.assignment_id,
            "assignment_label": render_training_label(result.presentation),
            "training_run_id": result.training_run_id,
            "evaluation_run_id": result.evaluation_run_id,
            "split_artifact_path": result.split_artifact_path,
        }
        for result in results
    }


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
    presentation: TrainingPresentation,
    training_run_id: str,
    split_artifact_path: str,
) -> None:
    """Set bookkeeping tags/params dlkit has no concept of on a completed eval run."""
    if run_id is None:
        return
    tags = build_evaluation_tags(
        assignment_id=assignment_id,
        presentation=presentation,
        source_training_run_id=training_run_id,
        split_artifact=split_artifact_path,
    ).as_mlflow_tags()
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    client.log_param(run_id, "source_training_run_id", training_run_id)
    client.log_param(run_id, "split_artifact_path", split_artifact_path)


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
        presentation=prepared.presentation,
        training_run_id=prepared.training_run_id,
        split_artifact_path=prepared.split_artifact_path,
    )
    return EvaluationRunResult(
        assignment_id=prepared.assignment.id,
        presentation=prepared.presentation,
        training_run_id=prepared.training_run_id,
        evaluation_run_id=result.mlflow_run_id,
        metrics=dict(result.metrics),
        split_file=prepared.split_file,
        split_artifact_path=prepared.split_artifact_path,
        figures_dir=figures_dir,
    )


def _run_prepared_eval_sweep(
    *,
    preparation: EvaluationPreparationBatch,
    experiment_name: str,
    parent_run: NamedRun,
    parent_tags: dict[str, str],
) -> tuple[Any | None, list[EvaluationRunResult], bool]:
    if not preparation.run_specs:
        return None, [], preparation.any_failed
    sweep_result = run_multirun_spec(
        MultiRunSpec(
            experiment_name=experiment_name,
            parent_run_name=run_name_text(parent_run),
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
                    "{} failed: {}",
                    render_training_label(prepared.presentation),
                    child_outcome.message,
                )
    return sweep_result, results, any_failed


def _resolve_eval_parent_run(
    *,
    sweep_result: Any | None,
    tracking_uri: str,
    artifact_uri: str | None,
    run: NamedRun,
    tags: dict[str, str],
    experiment_name: str,
) -> tuple[str, str]:
    match sweep_result:
        case None:
            parent_run_id = create_session_parent_run(
                tracking_uri=tracking_uri,
                artifact_uri=artifact_uri,
                run=run,
                tags=tags,
                experiment_name=experiment_name,
            )
            return parent_run_id, tracking_uri
        case _:
            resolved_tracking_uri = sweep_result.tracking_uri or tracking_uri
            return sweep_result.parent_run_id, resolved_tracking_uri


def eval_batch(
    cfg: CaseConfig,
    case_config_path: CaseConfigPath,
    settings: NeurallsSettings,
    output_root: Path | None = None,
    assignment_ids: Sequence[str] | None = None,
    tracking: TrackingServices = MLFLOW_TRACKING,
) -> EvaluationBatchResult:
    """Evaluate selected assignments from one case config as one dlkit multirun sweep."""
    configs_dir = case_config_path.config_dir
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

    presentations = training_presentations(
        cfg,
        config_dir=configs_dir,
        settings=settings,
        assignment_ids=[assignment.id for assignment in assignments],
        load_dataset=load_dataset_presentation,
    )
    base_output = resolve_base_output(cfg, settings, output_root, configs_dir)
    eval_mlflow_env = build_workflow_environment(
        tracking_uri=cfg.mlflow.tracking_uri,
        artifact_location=cfg.mlflow.artifacts_destination,
        default_output_root=base_output,
        config_dir=configs_dir,
    )
    mlflow_experiment_name = cfg.names.training

    with scoped_mlflow_environment(eval_mlflow_env.env):
        session_run, session_tags = build_session_run_spec(
            title=session_title(
                [
                    ComparisonScopePresentation(training_dataset(p.subject))
                    for p in presentations.values()
                ]
            ),
            notation=case_notation(cfg),
            case_config_path=case_config_path.path,
            experiment_name=mlflow_experiment_name,
            phase="session_evaluation",
        )
        session_tag_map = dict(session_tags.as_mlflow_tags())

        with tracking.lease_manager(eval_mlflow_env.tracking_uri) as artifact_leases:
            preparation = prepare_eval_run_specs(
                assignments=assignments,
                presentations=presentations,
                cfg=cfg,
                configs_dir=configs_dir,
                settings=settings,
                output_root=base_output,
                case_config_path=case_config_path.path,
                tracking=tracking,
                tracking_uri=eval_mlflow_env.tracking_uri,
                artifact_leases=artifact_leases,
            )
            sweep_result, results, any_failed = _run_prepared_eval_sweep(
                preparation=preparation,
                experiment_name=mlflow_experiment_name,
                parent_run=session_run,
                parent_tags=session_tag_map,
            )

        parent_run_id, tracking_uri = _resolve_eval_parent_run(
            sweep_result=sweep_result,
            tracking_uri=eval_mlflow_env.tracking_uri,
            artifact_uri=eval_mlflow_env.artifact_uri,
            run=session_run,
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
