"""Batch orchestration for comparison workflows."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from torchalg.utils.device import resolve_device

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
    load_master_config,
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
from neuralls.domain.presentation.titles import (
    summarized_session_title,
)
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.tracking.failure_reporting import MLflowWorkflowFailureSink
from neuralls.platform.tracking.method_sink import MLflowBenchmarkMethodSink
from neuralls.platform.tracking.metrics_logging import (
    setup_comparison_tracking,
)
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.platform.tracking.session_artifacts import SESSION_PHASE
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
        ComparisonBatchFailureContext(phase=phase, case_config_path=case_config_path.resolve()),
    )


def _run_session(
    batch: PreparedComparisonBatch,
    executions: Sequence[PreparedComparisonExecution],
    topology: ComparisonTopology,
    case_config_path: Path,
) -> list[ComparisonOutcome]:
    """Execute the batch inside one session parent run and project the workloads to outcomes."""
    pending = batch.items
    session_run, session_tags = build_session_run_spec(
        title=summarized_session_title(
            [item.title for item in executions],
            session_counts([item.planned for item in executions]),
        ),
        notation=batch_notation_policy(executions),
        case_config_path=case_config_path.resolve(),
        experiment_name=topology.experiment_name,
        phase=SESSION_PHASE,
    )
    failure_sink = MLflowWorkflowFailureSink()
    with mlflow.start_run(run_name=run_name_text(session_run), tags=session_tags.as_mlflow_tags()):
        with _batch_phase(failure_sink, case_config_path, ComparisonBatchPhase.WARMUP):
            warm_up_device(resolve_device())
        with _batch_phase(failure_sink, case_config_path, ComparisonBatchPhase.METHOD_WORKLOADS):
            workloads = execute_all(
                batch,
                method_sink=MLflowBenchmarkMethodSink(
                    batch_dataset_presentations(executions),
                    batch_solver_timing_contexts(executions),
                    batch_reference_selections(executions),
                ),
                failure_sink=failure_sink,
            )
        with _batch_phase(failure_sink, case_config_path, ComparisonBatchPhase.RESULT_PROJECTION):
            return outcomes_from_workloads(pending, workloads)


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
    master_cfg, _ = load_master_config(case_config_path, settings)
    if not master_cfg.comparisons and not master_cfg.comparison_sweeps:
        raise ValueError(
            "Case config must define at least one [[comparisons]] or [[comparison_sweeps]] entry."
        )

    topology = resolve_comparison_topology(case_config_path, settings)
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
            MlflowClient(tracking_uri=topology.model_store_tracking_uri),
            stack,
            force=params.force,
            reuse_comparisons=False,
        )
        partition = partition_pending(batch.items)
        if not partition.executions:
            return list(partition.decided)
        if has_complete_reusable_session(batch, topology, force=params.force):
            return [
                item
                if is_comparison_outcome(item)
                else ComparisonReused(comparison_id=item.entry.id, run=item.run_name)
                for item in batch.items
            ]
        return _run_session(batch, partition.executions, topology, case_config_path)
