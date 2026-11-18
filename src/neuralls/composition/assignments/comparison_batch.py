"""Batch orchestration for comparison workflows."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from torchalg.utils.device import resolve_device

from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.composition.assignments.comparison_execute import (
    batch_dataset_presentations,
    batch_notation_policy,
    batch_reference_selections,
    batch_solver_timing_contexts,
    execute_all,
    outcomes_from_workloads,
)
from neuralls.composition.assignments.comparison_prepare import (
    prepare_comparison_batch,
)
from neuralls.composition.assignments.comparison_prepared import (
    PreparedComparisonBatch,
    PreparedComparisonExecution,
    partition_pending,
)
from neuralls.composition.assignments.comparison_schedule import (
    has_complete_reusable_session,
)
from neuralls.composition.assignments.comparison_topology import (
    ComparisonTopology,
    failure_boundary,
    resolve_comparison_topology,
)
from neuralls.composition.comparison.models import (
    ComparisonOutcome,
    ComparisonParams,
    ComparisonReused,
    is_comparison_outcome,
)
from neuralls.composition.comparison.recipes import session_counts
from neuralls.composition.tracking.run_specs import (
    build_session_run_spec,
)
from neuralls.composition.tracking.services import MLFLOW_TRACKING, TrackingServices
from neuralls.domain.presentation.run_names import COMPARISON_SESSION_PHASE
from neuralls.domain.presentation.titles import (
    summarized_session_title,
)
from neuralls.platform.config.case_path import CaseConfigPath
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.tracking.comparison_runs import MethodRecordContext
from neuralls.platform.tracking.mlflow import ensure_experiment
from neuralls.platform.tracking.runtime_environment import collect_runtime_environment
from neuralls.shared.device import warm_up_device
from neuralls.shared.digest import digest_memo
from neuralls.shared.workflow_failures import (
    ComparisonBatchFailureContext,
    ComparisonBatchPhase,
    WorkflowFailureSink,
)


def _batch_phase(
    sink: WorkflowFailureSink, case_config_path: Path, phase: ComparisonBatchPhase
) -> AbstractContextManager[None]:
    """The failure boundary recording a blocking error of one batch phase."""
    return failure_boundary(
        sink,
        ComparisonBatchFailureContext(phase=phase, case_config_path=case_config_path),
    )


def _run_session(
    batch: PreparedComparisonBatch,
    executions: Sequence[PreparedComparisonExecution],
    topology: ComparisonTopology,
    experiment_id: str,
    case_config_path: Path,
    tracking: TrackingServices,
) -> list[ComparisonOutcome]:
    """Execute the batch inside one session parent run and project the workloads to outcomes."""
    pending = batch.items
    device = resolve_device()
    environment = collect_runtime_environment(device)
    session_run, session_tags = build_session_run_spec(
        title=summarized_session_title(
            [item.title for item in executions],
            session_counts([item.planned for item in executions]),
        ),
        notation=batch_notation_policy(executions),
        case_config_path=case_config_path,
        experiment_name=topology.experiment_name,
        phase=COMPARISON_SESSION_PHASE,
        revision=environment.revision,
    )
    runs = tracking.comparison_runs(
        topology.tracking_uri,
        experiment_id,
        MethodRecordContext(
            batch_dataset_presentations(executions),
            batch_solver_timing_contexts(executions),
            batch_reference_selections(executions),
        ),
    )
    with runs.session(session_run, session_tags.as_mlflow_tags()) as session:
        session.record_splits(batch.split_records)
        session.record_environment(environment)
        with _batch_phase(session.failures, case_config_path, ComparisonBatchPhase.WARMUP):
            warm_up_device(device)
        with _batch_phase(
            session.failures, case_config_path, ComparisonBatchPhase.METHOD_WORKLOADS
        ):
            workloads = execute_all(
                batch,
                session=session,
                training_runs=tracking.client(topology.model_store_tracking_uri),
                device=device,
            )
        with _batch_phase(
            session.failures, case_config_path, ComparisonBatchPhase.RESULT_PROJECTION
        ):
            return outcomes_from_workloads(pending, workloads)


def run_comparison_batch(
    case_config_path: CaseConfigPath,
    params: ComparisonParams,
    settings: NeurallsSettings,
    *,
    tracking: TrackingServices = MLFLOW_TRACKING,
) -> list[ComparisonOutcome]:
    """Load the case config from disk and run its comparison entries.

    The disk edge of `run_loaded_comparison_batch`.
    """
    master_cfg, _ = load_validated_case_config(case_config_path.path, settings)
    return run_loaded_comparison_batch(
        master_cfg, case_config_path, params, settings, tracking=tracking
    )


def run_loaded_comparison_batch(
    master_cfg: CaseConfig,
    case_config_path: CaseConfigPath,
    params: ComparisonParams,
    settings: NeurallsSettings,
    *,
    tracking: TrackingServices = MLFLOW_TRACKING,
) -> list[ComparisonOutcome]:
    """Run all comparison entries of an already-loaded case config.

    Every entry is resolved and reuse-checked first, with no MLflow run
    opened yet. Only if at least one entry genuinely needs execution does
    this open one MLflow session parent run for the whole batch — named
    ``"<case-name> | <timestamp>"``, mirroring training's session parent —
    with one direct method child per executing workload. If every entry is a
    cache hit, no MLflow run is created at all.

    Args:
        master_cfg: The validated case config.
        case_config_path: Where the case config lives; anchors relative paths and
            identifies the case. It is not read here.
        params: Comparison execution parameters (`force`).
        settings: Resolved runtime settings.
        tracking: Source of the MLflow clients, lease managers and reuse lookups.

    Returns:
        List of comparison outcomes, one per planned system, in plan order
        (sweep systems follow all [[comparisons]] systems).
    """
    if not master_cfg.comparisons and not master_cfg.comparison_sweeps:
        raise ValueError(
            "Case config must define at least one [[comparisons]] or [[comparison_sweeps]] entry."
        )

    case_path = case_config_path.path
    topology = resolve_comparison_topology(master_cfg, case_config_path, settings)
    experiment_id = ensure_experiment(
        topology.experiment_name,
        MlflowPaths(tracking_uri=topology.tracking_uri, artifact_uri=topology.artifact_location),
    )

    with contextlib.ExitStack() as stack:
        stack.enter_context(digest_memo())
        batch = prepare_comparison_batch(
            master_cfg,
            topology,
            case_path,
            settings,
            tracking,
            stack,
        )
        partition = partition_pending(batch.items)
        if not partition.executions:
            return list(partition.decided)
        if has_complete_reusable_session(batch, topology, tracking, force=params.force):
            return [
                item
                if is_comparison_outcome(item)
                else ComparisonReused(comparison_id=item.entry.id, run=item.run_name)
                for item in batch.items
            ]
        return _run_session(
            batch, partition.executions, topology, experiment_id, case_path, tracking
        )
