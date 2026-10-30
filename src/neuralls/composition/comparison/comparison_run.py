"""One comparison run across all configured preconditioners — pure orchestration entry point."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch
from loguru import logger
from torchalg.preconditioners.base import Preconditioner
from torchalg.utils.device import resolve_device

from neuralls.composition.comparison._generation_cost import resolve_generation_cost
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    _bind_extra_inputs,
    _create_scheduled_preconditioners,
    _load_extra_inputs,
    capture_system_realizations,
)
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
    run_cg_comparison,
)
from neuralls.domain.solver.models.benchmark_events import (
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
from neuralls.domain.solver.models.cost import StageCost
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
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

    Mirrors the shape ``run_cg_comparison`` already produces for a solver
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
) -> dict[str, Preconditioner]:
    """Construct, `.setup()` and schedule-wrap one preconditioner — the timed unit of a build."""
    return _create_scheduled_preconditioners(
        preconditioner_configs=[cfg],
        matrix=matrix,
        base_preconditioners={cfg.name: service.create_preconditioner(matrix, cfg)},
    )


def _build_scheduled_and_bind(
    service: PreconditionerService,
    matrix: torch.Tensor,
    cfg: PreconditionerConfig,
    staged_inputs: tuple[np.ndarray, ...],
) -> dict[str, Preconditioner]:
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
        offline_cost: One-off generation cost attached to every result, if the method has one.
        observer: Tracks the workload's peak memory across setup and solves.
        method_text: The method as its log lines name it.
    """

    method: ResolvedMethod
    service: PreconditionerService
    params: SolverParams
    device: torch.device
    offline_cost: StageCost | None
    observer: WorkloadMemoryObserver
    method_text: str


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
        "Recoverable setup failure for {} on {}, matrix {}: {}: {}",
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
    scheduled: dict[str, Preconditioner],
    setup_event: MatrixSetupEvent,
    dataset_text: str,
) -> MatrixBundleResult:
    """Solve every right-hand side of one matrix with its single built preconditioner."""
    method, params = run.method, run.params
    realizations = tuple(
        capture_system_realizations(scheduled[method.name], PreconditionerName(method.name), bundle)
    )
    attempts: list[SolveAttempt] = []
    diagnostics: list[tuple[SystemId, CGComparisonResult]] = []
    for member in bundle.members:
        rhs = torch.as_tensor(member.rhs).to(device=run.device, dtype=matrix.dtype)
        x_exact = compute_reference_solution(
            matrix,
            rhs,
            rtol=params.rtol,
            margin=params.reference_precision_margin,
        )
        result = run_cg_comparison(
            matrix,
            rhs,
            preconditioners=scheduled,
            x_exact=x_exact,
            rtol=params.rtol,
            atol=params.atol,
            maxiter=params.max_iterations,
            m_max=params.m_max,
            reference_precision_margin=params.reference_precision_margin,
            timing=params.timing,
        )[method.config.name]
        if run.offline_cost is not None:
            result = replace(result, generation_cost=run.offline_cost)
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
                "Recoverable solve failure for {} on {}, matrix {}, RHS {}: {}",
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
            )
        )
        del rhs, x_exact
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
        matrix = torch.as_tensor(bundle.matrix).to(run.device)
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
    datasets: Mapping[DatasetId, DatasetPresentation],
    notation: NotationPolicy,
    params: SolverParams,
    settings: NeurallsSettings,
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
    offline_cost = resolve_generation_cost(method.config, settings=settings)
    method_text = render_method_label(prepared.configured, notation)

    with CheckpointStore() as checkpoints:
        run = _MethodRun(
            method=method,
            service=PreconditionerService(FS_FIT_DATA, checkpoints=checkpoints),
            params=params,
            device=device,
            offline_cost=offline_cost,
            observer=observer,
            method_text=method_text,
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
        generation_cost=offline_cost,
        workload_peak_memory=MeasuredPeakMemory(observer.finish()),
        config_digest=method.identity_digest or canonical_digest(method.config),
        method_config=canonical_method_config(method.config),
        presentation=ExecutedMethod(prepared, tuple(realizations)),
        plot_styles={
            str(method.name): PreconditionerPlotStyle(
                presentation=prepared,
                family=preconditioner_family(method.config),
                color_key=color_key,
                marker_key=marker_key,
            )
        },
    )
