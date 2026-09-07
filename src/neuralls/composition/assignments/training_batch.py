"""Batch training workflow: prepare/reuse-check every assignment, then train them as one dlkit multirun sweep.

This module provides the main orchestration functions for running assignments
(one job run on one dataset):
- `run_assignment()`: Reuse-check one assignment's already-generated dataset
  and either report a cache-hit success or prepare it for the sweep.
- `run_assignment_sweep()`: Run every assignment from one case config as a
  single dlkit multirun sweep — the one training-stage implementation shared
  by both the `neuralls train` and `neuralls run` CLI commands.
- `write_metric_report()`: Build and upload the sweep's aggregate metric plot
  and label map, used by `neuralls train`'s CLI reporting.

Architecture:
    1. Data generation is a separate, earlier stage (`generation.multi_generation
       .generate_batch`) — this module assumes it already ran and only reads
       what's on disk. It never generates a dataset itself.
    2. Model training (with checkpoint detection) - dlkit's run_multirun_spec()
    3. Solver comparison (separate) - compare_preconditioners() from compare module

Note:
    This orchestrator does NOT handle dataset generation or solver comparisons.
    Use `composition.assignments.case_pipeline.run_case_pipeline()`, which
    composes `generate_batch()`, this sweep, and `run_comparison_batch()` as
    three independent stages.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dlkit.common import ChildFailure, ChildSuccess
from dlkit.engine.workflows.multi_run import MultiRunSpec, RunSpec
from dlkit.interfaces.api import run_multirun_spec
from loguru import logger

from neuralls.application.models import AssignmentResult, AssignmentSweepResult
from neuralls.composition.assignments.assembler import (
    load_assignment_batch,
    load_validated_case_config,
)
from neuralls.composition.assignments.training import (
    PreparedTraining,
    cleanup_prepared_training,
    finalize_prepared_training,
    prepare_training_settings,
    to_run_spec,
)
from neuralls.composition.tracking.run_specs import build_session_run_spec
from neuralls.platform.caching import compute_dataset_fingerprint
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.plots import plot_metric_comparison
from neuralls.platform.storage.dataset_readers import resolve_dataset_artifacts
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import (
    build_workflow_environment,
    finalize_session_parent_run,
)
from neuralls.platform.tracking.mlflow_client import (
    fetch_mlflow_metrics,
    find_successful_run,
    log_batch_artifacts_to_mlflow,
)


def run_assignment(
    *,
    settings: NeurallsSettings,
    job_config_path: Path,
    data_config_path: Path,
    output_root: Path,
    force: bool,
    max_epochs: int | None = None,
    assignment_id: str,
    assignment_display_name: str,
    dataset_registry_id: str | None = None,
    dataset_display_name: str | None = None,
    job_registry_id: str | None = None,
    job_display_name: str | None = None,
    mlflow_experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> AssignmentResult | PreparedTraining:
    """Reuse-check one assignment's already-generated dataset and prepare it for the sweep.

    Assumes dataset generation is a separate, earlier stage
    (`generation.multi_generation.generate_batch`) that already ran — this
    function only reads what's on disk; it never generates anything. If the
    dataset genuinely doesn't exist, resolving its artifacts below fails with
    a clear "re-run the generation pipeline" error.

    This function performs everything a dlkit multirun sweep child can't do for
    itself, ahead of `run_assignment_sweep()` building that sweep:
    1. Load data configuration from TOML file
    2. Resolve the already-generated dataset's artifacts (manifest + .npy + sparse pack)
    3. Check MLflow for an already-completed run of this assignment, keyed on both
       assignment_id and a fingerprint of the dataset's files — a dataset
       regenerated since that run no longer matches, so training reruns automatically
       (skip if a match is found and force=False)
    4. If training is needed, resolve dlkit settings via `prepare_training_settings()`

    Args:
        job_config_path: Path to a job configuration TOML (e.g., /path/to/job.toml)
        data_config_path: Path to a dataset configuration TOML (e.g., /path/to/dataset.toml)
        output_root: Root directory for all assignment outputs
        force: If True, retrain even if a completed run already exists. If False, reuse it.
        tracking_uri: MLflow tracking URI used to check for a prior completed run.

    Returns:
        An `AssignmentResult` with status='Success' (a completed MLflow run already
        exists and `force=False`) or status='Failed' (the dataset was never
        generated, the reuse check, or dlkit settings preparation raised).
        Otherwise a `PreparedTraining` ready to become one child of the
        training sweep in `run_assignment_sweep()`.

    Note:
        For solver comparison, use compare_preconditioners() after training completes.
        This function only reuse-checks and prepares models for training.

    Example:
        >>> outcome = run_assignment(
        ...     settings=settings,
        ...     job_config_path=Path("/tmp/job.toml"),
        ...     data_config_path=Path("/tmp/dataset.toml"),
        ...     output_root=Path("output"),
        ...     force=False,
        ...     assignment_id="assignment-1",
        ...     assignment_display_name="Assignment 1",
        ... )
        >>> isinstance(outcome, AssignmentResult) and outcome.status
        'Success'
    """
    try:
        # Step 1: Load data configuration and resolve the already-generated
        # dataset's directory. Fails fast if the dataset config has no
        # resolvable identity — the resolved name itself isn't needed here
        # now that the checkpoint-dir computation that used it has moved to
        # MLflow (Step 2 below).
        data_cfg = load_data_config(data_config_path, settings)
        resolve_dataset_identity(data_cfg=data_cfg, config_path=data_config_path)
        if data_cfg.output.data_dir is None:
            data_cfg = data_cfg.model_copy(
                update={"output": data_cfg.output.with_data_dir(settings.processed_dir)}
            )
        if data_cfg.output.data_dir is None:
            raise ValueError("output.data_dir could not be resolved")
        data_dir = data_cfg.output.data_dir / data_cfg.id
        artifacts = resolve_dataset_artifacts(data_dir)
        missing = [
            str(p)
            for p in (artifacts.rhs.path, artifacts.solutions.path, artifacts.matrix.path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Required data files not found in {data_dir}:\n  - " + "\n  - ".join(missing)
            )
        dataset_hash = compute_dataset_fingerprint(
            (artifacts.matrix.path, artifacts.rhs.path, artifacts.solutions.path)
        )

        # Step 2: Check MLflow for a completed run of this exact assignment against
        # this exact dataset. A FINISHED run tagged with this assignment_id and a
        # matching dataset_hash is trusted as equivalent to training again — nothing
        # is reused locally, whatever needs the checkpoint later (comparison,
        # inference) resolves it independently through MLflow. A dataset regenerated
        # since that run changes dataset_hash, so it no longer matches and training
        # reruns automatically.
        existing_run_id = (
            None
            if force or mlflow_experiment_name is None or tracking_uri is None
            else find_successful_run(
                tracking_uri=tracking_uri,
                mlflow_experiment_name=mlflow_experiment_name,
                assignment_id=assignment_id,
                dataset_hash=dataset_hash,
            )
        )

        # Step 3: Skip training if a completed run already exists; otherwise resolve
        # this assignment's dlkit settings so the caller can add it to the sweep.
        if existing_run_id is not None:
            logger.info(f"Using existing MLflow run for assignment '{assignment_id}'")
            return AssignmentResult(
                assignment_id=assignment_id,
                assignment_display_name=assignment_display_name,
                status="Success",
                mlflow_run_id=existing_run_id,
            )

        return prepare_training_settings(
            config_path=job_config_path,
            data_config_path=data_config_path,
            settings=settings,
            output_root=output_root,
            max_epochs=max_epochs,
            assignment_id=assignment_id,
            assignment_display_name=assignment_display_name,
            dataset_registry_id=dataset_registry_id,
            dataset_display_name=dataset_display_name,
            job_registry_id=job_registry_id,
            job_display_name=job_display_name,
            mlflow_experiment_name=mlflow_experiment_name,
            batched=True,
            extra_tags={"dataset_hash": dataset_hash},
        )
    except Exception as exc:  # noqa: BLE001
        # Broad by design: one assignment's failure (including dlkit-internal
        # errors, e.g. a leaked MLflow run from a prior search job) must never
        # abort the rest of the batch.
        logger.error(f"Assignment {assignment_id} failed: {exc}")
        return AssignmentResult(
            assignment_id=assignment_id,
            assignment_display_name=assignment_display_name,
            status="Failed",
            error=str(exc),
        )


def _finalize_assignment_child(
    *,
    prepared: PreparedTraining,
    execution_result: object,
) -> AssignmentResult:
    """Finalize one successful multirun child: make its checkpoint durable in MLflow.

    Converts a durability failure into a Failed `AssignmentResult` rather than
    raising — mirrors `run_assignment()`'s original per-assignment failure
    isolation, which used to cover the full train+finalize path via `train_model()`.

    Args:
        prepared: This assignment's prepared training inputs.
        execution_result: The multirun child's dispatch result.

    Returns:
        `AssignmentResult` with status='Success' or 'Failed'.
    """
    spec = prepared.assignment.spec
    try:
        run_id, _ = finalize_prepared_training(prepared, execution_result)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Assignment {spec.assignment_id} failed: {exc}")
        return AssignmentResult(
            assignment_id=spec.assignment_id,
            assignment_display_name=spec.assignment_display_name,
            status="Failed",
            error=str(exc),
        )
    logger.info(f"Training complete: run {run_id}")
    return AssignmentResult(
        assignment_id=spec.assignment_id,
        assignment_display_name=spec.assignment_display_name,
        status="Success",
        mlflow_run_id=run_id,
    )


def run_assignment_sweep(
    case_config_path: Path,
    settings: NeurallsSettings | None = None,
    *,
    force: bool = False,
    max_epochs: int | None = None,
) -> AssignmentSweepResult:
    """Run training for every assignment defined in one case config.

    Assumes dataset generation is a separate, earlier stage — see the module
    docstring. This is the main orchestrator for running multiple assignments:
    model training (with MLflow-backed reuse unless force=True), dispatched
    as one dlkit multirun sweep across every assignment that needs it.

    Assignments whose MLflow reuse check finds a completed run are resolved
    immediately without joining the sweep. Failed assignments — whether the
    failure happened during dataset resolution/preparation or during the
    sweep itself — don't stop the batch; each returns a result with status.

    Args:
        case_config_path: Path to a case config defining all assignments
        force: If True, retrain all models even if a completed run already exists

    Returns:
        `AssignmentSweepResult` with one `AssignmentResult` per assignment
        (success or failure, in the same order as the case config's
        `[[assignments]]`), plus the sweep's MLflow session parent run
        identity for aggregate reporting (`write_metric_report`).

    Note:
        For dataset generation and solver comparison, use
        `composition.assignments.case_pipeline.run_case_pipeline()`, which
        composes `generate_batch()`, this sweep, and `run_comparison_batch()`:
        >>> from neuralls.composition.assignments.case_pipeline import run_case_pipeline
        >>> run_case_pipeline(case_config_path, settings)

    Example:
        >>> sweep = run_assignment_sweep(
        ...     Path("/tmp/case.toml"),
        ...     force=False,
        ... )
        >>> success_count = sum(1 for r in sweep.results if r.status == "Success")
        >>> print(f"{success_count}/{len(sweep.results)} assignments succeeded")
    """
    settings = require_settings(settings, case_config_path=case_config_path)

    # Load all assignment definitions from the case config
    batch = load_assignment_batch(case_config_path, settings)
    assignments = batch.assignments

    logger.info(f"Training {len(assignments)} assignments from {case_config_path}")

    # Resolve the case-level MLflow topology once. Every assignment needing
    # training joins one dlkit multirun sweep below — dlkit owns that sweep's
    # parent run's lifecycle and per-child failure isolation natively.
    cfg, _ = load_validated_case_config(case_config_path, settings)
    training_mlflow_env = build_workflow_environment(
        tracking_uri=cfg.mlflow.tracking_uri,
        artifact_location=cfg.mlflow.artifacts_destination,
        default_output_root=batch.output_root,
    )
    mlflow_experiment_name = cfg.names.training

    results_by_id: dict[str, AssignmentResult] = {}
    prepared_by_child_id: dict[str, PreparedTraining] = {}
    run_specs: list[RunSpec] = []
    any_failed = False

    with scoped_mlflow_environment(training_mlflow_env.env):
        session_run_name, session_tags = build_session_run_spec(
            case_config_path=case_config_path.resolve(),
            experiment_name=mlflow_experiment_name,
            phase="session_training",
        )

        # Phase 1: Generate/cache each assignment's dataset and run its MLflow
        # reuse check. Cache hits resolve immediately; everything else is
        # prepared as one child of the sweep below.
        for assignment in assignments:
            spec = assignment.spec
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Assignment: {spec.assignment_display_name}")
            logger.info(f"  Job: {spec.job_display_name or spec.job_config_path.stem}")
            logger.info(
                f"  Dataset: {spec.dataset_display_name or assignment.workspace.dataset_id}"
            )
            logger.info(f"{'=' * 60}")

            outcome = run_assignment(
                settings=settings,
                job_config_path=spec.job_config_path,
                data_config_path=spec.data_config_path,
                output_root=batch.output_root,
                force=force,
                max_epochs=max_epochs,
                assignment_id=spec.assignment_id,
                assignment_display_name=spec.assignment_display_name,
                dataset_registry_id=spec.dataset_id,
                dataset_display_name=spec.dataset_display_name,
                job_registry_id=spec.job_id,
                job_display_name=spec.job_display_name,
                mlflow_experiment_name=mlflow_experiment_name,
                tracking_uri=training_mlflow_env.tracking_uri,
            )
            match outcome:
                case PreparedTraining():
                    prepared_by_child_id[spec.assignment_id] = outcome
                    run_specs.append(to_run_spec(outcome))
                case AssignmentResult():
                    results_by_id[spec.assignment_id] = outcome
                    if not outcome.is_success:
                        any_failed = True

        # Phase 2: Dispatch every assignment that needs training as one sweep.
        sweep_result = None
        if run_specs:
            sweep_result = run_multirun_spec(
                MultiRunSpec(
                    experiment_name=mlflow_experiment_name,
                    parent_run_name=session_run_name,
                    parent_tags=dict(session_tags.as_mlflow_tags()),
                    failure_policy="continue",
                    children=tuple(run_specs),
                )
            )
            for child_outcome in sweep_result.children:
                prepared = prepared_by_child_id[child_outcome.child_id]
                try:
                    match child_outcome:
                        case ChildSuccess():
                            result = _finalize_assignment_child(
                                prepared=prepared,
                                execution_result=child_outcome.result,
                            )
                            results_by_id[child_outcome.child_id] = result
                            if not result.is_success:
                                any_failed = True
                        case ChildFailure():
                            any_failed = True
                            logger.error(
                                f"Assignment '{child_outcome.child_id}' failed: {child_outcome.message}"
                            )
                            results_by_id[child_outcome.child_id] = AssignmentResult(
                                assignment_id=child_outcome.child_id,
                                assignment_display_name=prepared.resolved_assignment_display_name,
                                status="Failed",
                                error=child_outcome.message,
                            )
                finally:
                    cleanup_prepared_training(prepared)

        if sweep_result is not None:
            finalize_session_parent_run(
                tracking_uri=sweep_result.tracking_uri or training_mlflow_env.tracking_uri,
                run_id=sweep_result.parent_run_id,
                status="FAILED" if any_failed else "FINISHED",
            )

    return AssignmentSweepResult(
        results=[results_by_id[assignment.spec.assignment_id] for assignment in assignments],
        tracking_uri=(
            (sweep_result.tracking_uri or training_mlflow_env.tracking_uri)
            if sweep_result is not None
            else None
        ),
        parent_run_id=sweep_result.parent_run_id if sweep_result is not None else None,
    )


def _build_label_map(results: list[AssignmentResult]) -> dict[str, dict[str, str | None]]:
    """Build a mapping from assignment_id to its full identity, for the batch label map."""
    return {
        result.assignment_id: {
            "assignment_id": result.assignment_id,
            "assignment_display_name": result.assignment_display_name,
            "mlflow_run_id": result.mlflow_run_id,
        }
        for result in results
    }


def write_metric_report(sweep_result: AssignmentSweepResult, *, metric: str) -> bool:
    """Build the sweep's aggregate metric plot and label map, and upload both to MLflow.

    Stages the bar chart and label map in a scratch directory and uploads them
    via `log_batch_artifacts_to_mlflow` onto the sweep's session parent run —
    nothing is left on local disk, matching every other artifact this sweep
    produces, which all live under MLflow rather than a local output directory.

    Args:
        sweep_result: Completed `run_assignment_sweep()` result.
        metric: Metric key to plot across assignments.

    Returns:
        True if a comparison plot was produced (metric present for >=1 result).
    """
    if sweep_result.tracking_uri is None or sweep_result.parent_run_id is None:
        logger.warning(
            "No MLflow session parent run for this sweep; skipping metric report upload."
        )
        return False

    labels: list[str] = []
    values: list[float] = []
    missing: list[str] = []

    for result in sweep_result.results:
        metrics = (
            fetch_mlflow_metrics(result.mlflow_run_id, sweep_result.tracking_uri)
            if result.mlflow_run_id is not None
            else {}
        )
        if metric in metrics:
            labels.append(result.assignment_id)
            values.append(metrics[metric])
            continue
        missing.append(f"{result.assignment_id} ({result.assignment_display_name})")

    if missing:
        logger.warning("Metric '{}' missing for assignments: {}", metric, ", ".join(missing))

    legend = {
        result.assignment_id: (
            f"{result.assignment_display_name} (run: {result.mlflow_run_id})"
            if result.mlflow_run_id
            else result.assignment_display_name
        )
        for result in sweep_result.results
        if result.assignment_id in labels
    }

    plot_name = f"batch_metric_{metric.replace('/', '_')}.png"
    with tempfile.TemporaryDirectory() as tmp:
        work_root = Path(tmp)
        plotted = bool(labels)
        if plotted:
            plot_metric_comparison(
                labels=labels,
                values=values,
                metric_name=metric,
                legend=legend,
                save_path=work_root / plot_name,
            )

        (work_root / "batch_training_labels.json").write_text(
            json.dumps(_build_label_map(sweep_result.results), indent=2),
            encoding="utf-8",
        )

        log_batch_artifacts_to_mlflow(
            tracking_uri=sweep_result.tracking_uri,
            run_id=sweep_result.parent_run_id,
            work_root=work_root,
            flat_files=(plot_name, "batch_training_labels.json"),
        )

    return plotted
