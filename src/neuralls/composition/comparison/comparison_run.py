"""One comparison run across all configured preconditioners — pure orchestration entry point."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch
from loguru import logger
from mlflow.tracking import MlflowClient
from torchalg.preconditioners.base import Preconditioner
from torchalg.utils.device import resolve_device

from neuralls.composition.comparison._offline_cost import resolve_offline_cost
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    _bind_extra_inputs,
    _load_extra_inputs,
    _schedule_preconditioner,
    capture_system_realizations,
)
from neuralls.composition.comparison.method_data_origin import MethodDataOrigin, origin_evidence
from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    MatrixWorkload,
    ResolvedMethod,
    canonical_method_config,
)
from neuralls.composition.preconditioners.checkpoint_store import CheckpointStore
from neuralls.composition.preconditioners.default_services import FS_FIT_DATA
from neuralls.domain.presentation.methods import (
    ExecutedMethod,
    PreparedMethodPresentation,
    SystemRealization,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.comparison import (
    _to_numpy,
    compute_reference_solution,
    prepare_reference,
    solve_system,
)
from neuralls.domain.solver.error_metrics import reference_target
from neuralls.domain.solver.models.benchmark_events import (
    ExactSolutionUnavailable,
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SetupFailed,
    SolveAttempt,
    SolveConverged,
    SolveFailed,
    SystemTag,
    UnknownPeakMemory,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.config import SolverParams
from neuralls.domain.solver.models.cost import Known, OfflineStages, Unavailable
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.platform.config.models.preconditioner import PreconditionerConfig, PreconditionerType
from neuralls.platform.config.models.preconditioner_family import (
    pod2g_style_keys,
    preconditioner_family,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.tracking.run_names import render_dataset_label, render_method_label
from neuralls.shared.device import (
    FailedTimedRuns,
    WorkloadMemoryObserver,
    release_device_memory,
    repeat_timed_outcome,
)
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, PreconditionerName, SetupEventId, SystemId


def _breakdown_result(name: str, *, rhs: torch.Tensor, error: str) -> CGComparisonResult:
    """Stand-in result for a preconditioner that failed to build (e.g. IC(0) breakdown).

    Mirrors the shape ``solve_system`` already produces for a solver
    failure (``domain/solver/comparison.py``), so downstream reporting and
    MLflow logging — which already branch on ``result.error``/``.breakdown``
    — handle this the same way, with no new special case.

    Args:
        name: Preconditioner config name.
        rhs: Right-hand side vector, for ``rhs_norm`` and placeholder shape.
        error: The construction failure message.

    Returns:
        CGComparisonResult: ``converged=False``, ``breakdown=True``, zero iterations.
    """
    zeros = _to_numpy(torch.zeros_like(rhs))
    return CGComparisonResult(
        x=zeros,
        converged=False,
        iterations=0,
        residual=float("inf"),
        residual_abs=float("inf"),
        residual_history_rel=[],
        residual_history_abs=[],
        preconditioner=name,
        initial_guess=zeros,
        exact_error=None,
        rhs_norm=float(torch.linalg.vector_norm(rhs)),
        breakdown=True,
        error=f"Preconditioner failed: {error}",
    )


def _build_scheduled(
    service: PreconditionerService, matrix: torch.Tensor, cfg: PreconditionerConfig
) -> Preconditioner:
    """Construct, `.setup()` and schedule-wrap one preconditioner — the timed unit of a build."""
    return _schedule_preconditioner(cfg, matrix, service.create_preconditioner(matrix, cfg))


def _build_scheduled_and_bind(
    service: PreconditionerService,
    matrix: torch.Tensor,
    cfg: PreconditionerConfig,
    staged_inputs: tuple[np.ndarray, ...],
) -> Preconditioner:
    """Build one method and perform required conversion/binding in setup time."""
    scheduled = _build_scheduled(service, matrix, cfg)
    _bind_extra_inputs(scheduled, matrix, staged_inputs)
    return scheduled


@dataclass(frozen=True, slots=True)
class _MethodRun:
    """What stays fixed while one method executes over every matrix bundle.

    Attributes:
        method: The method being executed.
        service: Builds and preloads the method's preconditioner for each matrix.
        params: Solver and timing parameters.
        device: Device the matrices and solves live on.
        offline_cost: One-off generation and training cost attached to every result.
        observer: Tracks the workload's peak memory across setup and solves.
        method_text: The method as its log lines name it.
        reference_target_rel: Relative residual every reference solve of the run must reach.
    """

    method: ResolvedMethod
    service: PreconditionerService
    params: SolverParams
    device: torch.device
    offline_cost: OfflineStages
    observer: WorkloadMemoryObserver
    method_text: str
    reference_target_rel: float


@dataclass(frozen=True, slots=True)
class MatrixBundleResult:
    """Everything one matrix contributed to a method's workload.

    Attributes:
        setup_event: The matrix's single setup, completed or failed.
        attempts: One solve attempt per right-hand side; empty when setup failed.
        diagnostics: Each system's solver (or stand-in breakdown) result.
        realizations: Facts read from the built preconditioner; empty when setup failed.
    """

    setup_event: MatrixSetupEvent
    attempts: tuple[SolveAttempt, ...]
    diagnostics: tuple[tuple[SystemId, CGComparisonResult], ...]
    realizations: tuple[SystemRealization, ...]


@dataclass(frozen=True, slots=True)
class _SetupTiming:
    """What a setup measured before failing (all zero when it failed before being timed)."""

    seconds: float = 0.0
    samples_seconds: tuple[float, ...] = ()
    peak_memory_bytes: int = 0


def _setup_failed(
    run: _MethodRun,
    bundle: MatrixWorkload,
    event_id: SetupEventId,
    *,
    error: Exception,
    dataset_text: str,
    timing: _SetupTiming,
) -> MatrixBundleResult:
    """Record a recoverable setup failure: one failed event and a breakdown for every system.

    Args:
        run: The method run the matrix belongs to.
        bundle: The matrix whose setup failed.
        event_id: Identity of the matrix's setup event.
        error: What failed.
        dataset_text: The dataset as log lines name it.
        timing: What the timed setup measured before failing.
    """
    logger.warning(
        "Recoverable setup failure for {} on comparison matrix set {}, matrix {}: {}: {}",
        run.method_text,
        dataset_text,
        int(bundle.matrix_id),
        type(error).__name__,
        error,
    )
    return MatrixBundleResult(
        setup_event=MatrixSetupEvent(
            event_id=event_id,
            dataset=bundle.dataset,
            matrix_id=bundle.matrix_id,
            method=run.method.name,
            setup_s=timing.seconds,
            samples_s=timing.samples_seconds,
            peak_memory=MeasuredPeakMemory(timing.peak_memory_bytes),
            outcome=SetupFailed(f"{type(error).__name__}: {error}"),
        ),
        attempts=(),
        diagnostics=tuple(
            (
                member.tag.system_id,
                _breakdown_result(
                    run.method.name,
                    rhs=torch.as_tensor(member.rhs),
                    error=f"setup failed: {error}",
                ),
            )
            for member in bundle.members
        ),
        realizations=(),
    )


def _solve_bundle(
    run: _MethodRun,
    bundle: MatrixWorkload,
    matrix: torch.Tensor,
    preconditioner: Preconditioner,
    setup_event: MatrixSetupEvent,
    dataset_text: str,
) -> MatrixBundleResult:
    """Solve every right-hand side of one matrix with its single built preconditioner."""
    method, params = run.method, run.params
    realizations = tuple(
        capture_system_realizations(preconditioner, PreconditionerName(method.name), bundle)
    )
    attempts: list[SolveAttempt] = []
    diagnostics: list[tuple[SystemId, CGComparisonResult]] = []
    # Factored from the stored matrix, not the method's `matrix`, so every method of this
    # matrix compares against the same x* whatever representation it solves with.
    factored = prepare_reference(bundle.matrix)
    if isinstance(factored, ExactSolutionUnavailable):
        logger.warning(
            "Exact-error metrics skipped for {} on comparison matrix set {}, matrix {}: {}",
            run.method_text,
            dataset_text,
            int(bundle.matrix_id),
            factored.reason,
        )
    for member in bundle.members:
        rhs = torch.as_tensor(member.rhs).to(device=run.device, dtype=matrix.dtype)
        reference = compute_reference_solution(factored, rhs, target_rel=run.reference_target_rel)
        result = solve_system(
            matrix,
            rhs,
            preconditioner,
            name=method.config.name,
            reference=reference,
            params=params,
        )
        release_device_memory()
        if isinstance(reference, ExactSolutionUnavailable) and reference is not factored:
            # A failed factorization was reported once above; this is one RHS's own failure.
            logger.warning(
                "Exact-error metrics skipped for {} on comparison matrix set {}, matrix {}, "
                "RHS {}: {}",
                run.method_text,
                dataset_text,
                int(bundle.matrix_id),
                member.tag.rhs_ordinal,
                reference.reason,
            )
        result = replace(result, offline_cost=run.offline_cost)
        diagnostics.append((member.tag.system_id, result))
        solve_peak = result.solve_peak_memory_bytes
        if solve_peak is not None:
            run.observer.observe_operation(solve_peak)
        outcome = (
            SolveConverged(result.iterations, result.residual)
            if result.converged
            else SolveFailed(result.iterations, result.error or "solver did not converge")
        )
        if isinstance(outcome, SolveFailed):
            logger.warning(
                "Recoverable solve failure for {} on comparison matrix set {}, matrix {}, RHS {}: {}",
                run.method_text,
                dataset_text,
                int(bundle.matrix_id),
                member.tag.rhs_ordinal,
                outcome.reason,
            )
        attempts.append(
            SolveAttempt(
                setup_event_id=setup_event.event_id,
                system_id=member.tag.system_id,
                dataset=bundle.dataset,
                recipe=member.tag.recipe,
                matrix_id=bundle.matrix_id,
                rhs_ordinal=member.tag.rhs_ordinal,
                method=method.name,
                solve_s=result.solve_time_seconds or 0.0,
                samples_s=result.solve_time_samples,
                peak_memory=(
                    MeasuredPeakMemory(solve_peak)
                    if solve_peak is not None
                    else UnknownPeakMemory()
                ),
                outcome=outcome,
                residual=result.true_residual,
                exact_reference=result.exact_reference,
                repeats=result.timing_agreement,
            )
        )
        del rhs, reference
    return MatrixBundleResult(
        setup_event=setup_event,
        attempts=tuple(attempts),
        diagnostics=tuple(diagnostics),
        realizations=realizations,
    )


def _execute_on_device(
    run: _MethodRun, bundle: MatrixWorkload, matrix: torch.Tensor, dataset_text: str
) -> MatrixBundleResult:
    """Build the method once for ``matrix`` (timed), then solve every right-hand side."""
    method = run.method
    staged_inputs = _load_extra_inputs(bundle.matrix_path, bundle.matrix_index)
    event_id = SetupEventId(f"{method.name}:{bundle.dataset}:{int(bundle.matrix_id)}")
    try:
        run.service.preload(matrix, method.config)
    except Exception as exc:  # noqa: BLE001
        return _setup_failed(
            run, bundle, event_id, error=exc, dataset_text=dataset_text, timing=_SetupTiming()
        )
    timed_setup = repeat_timed_outcome(
        lambda: _build_scheduled_and_bind(run.service, matrix, method.config, staged_inputs),
        run.params.timing,
        device=run.device,
    )
    run.observer.observe_operation(timed_setup.peak_memory_bytes)
    if isinstance(timed_setup, FailedTimedRuns):
        return _setup_failed(
            run,
            bundle,
            event_id,
            error=timed_setup.error,
            dataset_text=dataset_text,
            timing=_SetupTiming(
                timed_setup.seconds, timed_setup.samples_seconds, timed_setup.peak_memory_bytes
            ),
        )
    setup_event = MatrixSetupEvent(
        event_id=event_id,
        dataset=bundle.dataset,
        matrix_id=bundle.matrix_id,
        method=method.name,
        setup_s=timed_setup.seconds,
        samples_s=timed_setup.samples_seconds,
        peak_memory=MeasuredPeakMemory(timed_setup.peak_memory_bytes),
        outcome=SetupCompleted(),
    )
    return _solve_bundle(run, bundle, matrix, timed_setup.value, setup_event, dataset_text)


def _execute_matrix_bundle(
    run: _MethodRun, bundle: MatrixWorkload, dataset_text: str
) -> MatrixBundleResult:
    """Execute one matrix and release all device state on every exit path.

    Realized facts are read from the built preconditioner right after its one setup and
    returned as plain values; only those outlive the release below.
    """
    try:
        matrix = bundle.matrix.to(run.device)
        # TODO(investigate): neural methods get a dense matrix for setup *and* solve. The
        # predictor needs dense input, but PCG's own `A @ p` does not have to be dense.
        # Until settled: neural solve-only time includes an O(n^2) matvec where classical
        # methods pay O(nnz), and the `to_dense()` below is not timed. The type set repeats
        # `_DENSIFY_FOR_CSR` in composition/preconditioners/builders.py.
        if run.method.config.type in {PreconditionerType.NEURAL, PreconditionerType.NEURAL_AMG}:
            matrix = matrix.to_dense() if matrix.layout is torch.sparse_csr else matrix
        try:
            return _execute_on_device(run, bundle, matrix, dataset_text)
        finally:
            del matrix
    finally:
        release_device_memory()


def execute_method_workload(
    method: ResolvedMethod,
    bundles: Iterable[MatrixWorkload],
    *,
    prepared: PreparedMethodPresentation,
    origin: MethodDataOrigin,
    training_run_costs: Mapping[str, Known | Unavailable] | None = None,
    datasets: Mapping[DatasetId, DatasetPresentation],
    notation: NotationPolicy,
    params: SolverParams,
    settings: NeurallsSettings,
    training_runs: MlflowClient,
) -> CompletedMethodWorkload:
    """Execute one method over matrix bundles, keeping matrix state RHS-local.

    Dataset/checkpoint reads, reference solves, and reporting are outside the
    setup/solve timers.  Each matrix receives one fresh prepared method object;
    sibling RHS solves share that exact object.  The method-owned checkpoint
    store spans matrices and is released before this function returns.

    Facts read from each built preconditioner right after its one setup are kept with
    ``prepared``, which must belong to ``method``; a method whose every setup failed keeps
    none. ``datasets`` and ``notation`` are what its log lines call each dataset.
    """
    if prepared.method_id != method.name:
        raise ValueError(
            f"prepared presentation belongs to method {prepared.method_id!r}, not {method.name!r}"
        )
    device = resolve_device()
    observer = WorkloadMemoryObserver.start(device)
    planned_list: list[SystemTag] = []
    setup_events: list[MatrixSetupEvent] = []
    attempts: list[SolveAttempt] = []
    diagnostics: list[tuple[SystemId, CGComparisonResult]] = []
    realizations: list[SystemRealization] = []
    offline_cost = resolve_offline_cost(origin, runs=training_runs, run_costs=training_run_costs)
    method_text = render_method_label(prepared.configured, notation)
    target = reference_target(params.rtol, params.reference_precision_margin)
    if target.is_clamped:
        logger.warning(
            "Reference precision margin {:.1e} would ask below float64 machine precision at "
            "rtol={:.1e}; the reference solves of {} target {:.1e} relative residual instead.",
            params.reference_precision_margin,
            params.rtol,
            method_text,
            target.applied_rel,
        )

    with CheckpointStore() as checkpoints:
        run = _MethodRun(
            method=method,
            service=PreconditionerService(FS_FIT_DATA, checkpoints=checkpoints),
            params=params,
            device=device,
            offline_cost=offline_cost,
            observer=observer,
            method_text=method_text,
            reference_target_rel=target.applied_rel,
        )
        for bundle in bundles:
            planned_list.extend(member.tag for member in bundle.members)
            result = _execute_matrix_bundle(
                run, bundle, render_dataset_label(datasets[bundle.dataset], notation)
            )
            setup_events.append(result.setup_event)
            attempts.extend(result.attempts)
            diagnostics.extend(result.diagnostics)
            realizations.extend(result.realizations)
            del bundle

    color_key, marker_key = pod2g_style_keys(method.config)
    return CompletedMethodWorkload(
        method=PreconditionerName(method.name),
        planned=tuple(planned_list),
        setup_events=tuple(setup_events),
        solve_attempts=tuple(attempts),
        diagnostics=tuple(diagnostics),
        offline_cost=offline_cost,
        workload_peak_memory=MeasuredPeakMemory(observer.finish()),
        config_digest=method.identity_digest or canonical_digest(method.config),
        method_config=canonical_method_config(method.config),
        presentation=ExecutedMethod(prepared, tuple(realizations)),
        data_evidence=origin_evidence(origin),
        plot_styles={
            str(method.name): PreconditionerPlotStyle(
                presentation=prepared,
                family=preconditioner_family(method.config),
                color_key=color_key,
                marker_key=marker_key,
            )
        },
    )
