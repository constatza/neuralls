"""Execute a method-major schedule and project its workloads back to per-system outcomes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace

import mlflow

from neuralls.composition.assignments.comparison_prepared import (
    PendingComparison,
    PreparedComparisonBatch,
    PreparedComparisonExecution,
    described_failure,
)
from neuralls.composition.assignments.comparison_schedule import (
    MatrixGroup,
    load_matrix_workloads,
    method_major_schedule,
)
from neuralls.composition.assignments.comparison_topology import failure_boundary
from neuralls.composition.comparison.comparison_run import (
    execute_method_workload,
)
from neuralls.composition.comparison.models import (
    ComparisonCompleted,
    ComparisonOutcome,
    CompletedMethodWorkload,
    MatrixWorkload,
    ResolvedMethod,
    is_comparison_outcome,
)
from neuralls.composition.comparison.recipes import system_tag
from neuralls.composition.tracking.run_specs import (
    build_benchmark_method_run_spec,
)
from neuralls.domain.benchmarking.selection import (
    ReferenceSelection,
)
from neuralls.domain.presentation.methods import (
    MissingMethodPresentationError,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.comparison_summary import format_results_summary
from neuralls.domain.solver.models.benchmark_events import MeasuredPeakMemory, SetupCompleted
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.ports import BenchmarkMethodSink
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.platform.tracking.metrics_logging import (
    log_configured_method_facts,
)
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.shared.device import release_device_memory
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import (
    DatasetId,
    PreconditionerName,
    SystemId,
)
from neuralls.shared.workflow_failures import (
    ComparisonMatrixFailureContext,
    ComparisonMatrixPhase,
    ComparisonMethodFailureContext,
    ComparisonMethodPhase,
    WorkflowFailureSink,
)


def outcomes_from_workloads(
    pending: Sequence[PendingComparison],
    workloads: Sequence[CompletedMethodWorkload],
) -> list[ComparisonOutcome]:
    """Project canonical diagnostics back into the retained per-system API."""
    diagnostics: dict[str, dict[str, CGComparisonResult]] = {}
    for workload in workloads:
        for system_id, result in workload.diagnostics:
            result = project_setup_measurements(workload, system_id, result)
            diagnostics.setdefault(str(system_id), {})[str(workload.method)] = result

    outcomes: list[ComparisonOutcome] = []
    for item in pending:
        if is_comparison_outcome(item):
            outcomes.append(item)
            continue
        results = diagnostics.get(item.entry.id, {})
        if not results:
            outcomes.append(
                described_failure(item, RuntimeError("method workload produced no diagnostics"))
            )
            continue
        presentation = item.comparison_presentation
        try:
            summary = format_results_summary(
                results,
                {
                    name: presentation.methods.method(PreconditionerName(name)).compact
                    for name in results
                },
                presentation.notation,
            )
        except (MissingMethodPresentationError, KeyError) as exc:
            outcomes.append(described_failure(item, exc))
            continue
        outcomes.append(
            ComparisonCompleted(
                comparison_id=item.entry.id,
                run=item.run_name,
                warnings=item.resolved.warnings,
                summary=summary,
            )
        )
    return outcomes


def project_setup_measurements(
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


def _load_matrix_workloads_with_failure_context(
    method: ResolvedMethod,
    matrices: Sequence[MatrixGroup],
    *,
    failure_sink: WorkflowFailureSink,
    recorded_exception_ids: set[int],
) -> Iterator[MatrixWorkload]:
    """Load lazily while attributing a blocking read to its exact matrix."""
    for (dataset, matrix_id), systems in matrices:
        with failure_boundary(
            failure_sink,
            ComparisonMatrixFailureContext(
                phase=ComparisonMatrixPhase.INPUT_LOADING,
                method=method.name,
                dataset=dataset,
                matrix_id=matrix_id,
            ),
            recorded_exception_ids=recorded_exception_ids,
        ):
            bundle = next(iter(load_matrix_workloads((((dataset, matrix_id), systems),))))
        yield bundle


def _execute_method_run(
    method: ResolvedMethod,
    matrices: tuple[MatrixGroup, ...],
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
        with failure_boundary(
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
        with failure_boundary(
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


def execute_all(
    batch: PreparedComparisonBatch,
    *,
    method_sink: BenchmarkMethodSink,
    failure_sink: WorkflowFailureSink,
) -> tuple[CompletedMethodWorkload, ...]:
    """Execute a prepared batch in method order, each method inside its persisted child run."""
    schedule = method_major_schedule(batch.items)

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


def batch_notation_policy(executions: Sequence[PreparedComparisonExecution]) -> NotationPolicy:
    """The one notation every executed comparison of the batch uses; differing ones are ambiguous."""
    first, *rest = (item.cfg.notation_policy for item in executions)
    if any(policy != first for policy in rest):
        raise ValueError("every comparison of a batch must share one notation policy")
    return first


def batch_dataset_presentations(
    executions: Sequence[PreparedComparisonExecution],
) -> dict[DatasetId, DatasetPresentation]:
    """The presentation of each matrix dataset the batch runs on, from its manifest."""
    datasets: dict[DatasetId, DatasetPresentation] = {}
    for item in executions:
        dataset = DatasetId(item.entry.matrix_dataset)
        presented = item.presentation.scope.dataset
        if datasets.setdefault(dataset, presented) != presented:
            raise ValueError("one dataset must have one scientific presentation in a batch")
    return datasets


def batch_reference_selections(
    executions: Sequence[PreparedComparisonExecution],
) -> dict[SystemId, ReferenceSelection]:
    """Return explicit reference state for each planned system."""
    return {
        SystemId(str(system_tag(item.planned).system_id)): item.reference_selection
        for item in executions
    }


def batch_solver_timing_contexts(
    executions: Sequence[PreparedComparisonExecution],
) -> dict[SystemId, SolverTimingContext]:
    """Preserve the actual per-system solver and timing settings for figure metadata."""
    contexts: dict[SystemId, SolverTimingContext] = {}
    for item in executions:
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
