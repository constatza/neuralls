"""One comparison run across all configured preconditioners — pure orchestration entry point."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torchalg.preconditioners.base import Preconditioner
from torchalg.utils.device import resolve_device

from neuralls.composition.comparison._generation_cost import resolve_generation_cost
from neuralls.composition.comparison._linear_system import _load_linear_system
from neuralls.composition.comparison._plots import _generate_comparison_plots
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    SystemPreconditionerCache,
    _bind_extra_inputs,
    _create_scheduled_preconditioners,
    _load_extra_inputs,
    capture_system_realizations,
)
from neuralls.composition.comparison._presentation import build_comparison_source_context
from neuralls.composition.comparison.models import (
    ComparisonPaths,
    CompletedMethodWorkload,
    MatrixWorkload,
    PreconditionerComparisonEntry,
    ResolvedComparisonInput,
    ResolvedMethod,
    canonical_method_config,
)
from neuralls.composition.comparison.result_keys import validate_unique_preconditioner_keys
from neuralls.composition.preconditioners.checkpoint_store import CheckpointStore
from neuralls.composition.preconditioners.labels import (
    build_configured_preconditioner_label,
    build_preconditioner_labels,
    contextualize_preconditioner_label,
)
from neuralls.domain.presentation.methods import (
    ExecutedMethodPresentation,
    PreparedMethodPresentation,
    SystemRealization,
)
from neuralls.domain.presentation.planner import realize_method
from neuralls.domain.solver.comparison import (
    _to_numpy,
    compute_reference_solution,
    run_cg_comparison,
)
from neuralls.domain.solver.comparison_summary import (
    format_results_summary,
    summarize_best_combinations,
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
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle, plot_style_maps
from neuralls.domain.solver.models.recommendations import ComparisonRecommendations
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.platform.config.models.comparison import ComparisonGeneral
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
)
from neuralls.platform.config.models.preconditioner_family import preconditioner_family
from neuralls.platform.config.resolution import resolve_user_path
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.filesystem import ensure_dir
from neuralls.shared.device import (
    FailedTimedRuns,
    WorkloadMemoryObserver,
    release_device_memory,
    repeat_timed,
    repeat_timed_outcome,
)
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import PreconditionerName, SetupEventId, SystemId

type PreconditionerEvaluationMapper = Callable[
    [
        Callable[[PreconditionerConfig], PreconditionerComparisonEntry],
        Sequence[PreconditionerConfig],
    ],
    Iterable[PreconditionerComparisonEntry],
]


def _pod2g_style_keys(cfg: PreconditionerConfig) -> tuple[str | None, str | None]:
    """Extract (color_key, marker_key) for a POD-2G config, else (None, None).

    POD-2G comparisons routinely sweep many (fit dataset, snapshot weighting)
    combinations in one plot; the fit dataset drives color, the weighting
    scheme drives marker, so a dense sweep separates into distinct
    color/marker combinations instead of collapsing onto one family-wide
    marker+linestyle with only color-shade differences.

    Args:
        cfg: Preconditioner configuration to inspect.

    Returns:
        tuple[str | None, str | None]: ``(color_key, marker_key)``, both
        ``None`` for non-POD-2G configs (falls back to family-based styling).
    """
    if not isinstance(cfg, AMGPreconditionerConfig) or not isinstance(
        cfg.coarsening, PODCoarseningConfig | NeuralPODCoarseningConfig
    ):
        return None, None
    color_key = str(cfg.coarsening.dataset_dir)
    # Only PODCoarseningConfig carries a snapshot weighting scheme;
    # NeuralPODCoarseningConfig predicts snapshots from a checkpoint instead.
    marker_key = (
        cfg.coarsening.weighting.method if isinstance(cfg.coarsening, PODCoarseningConfig) else None
    )
    return color_key, marker_key


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


def _evaluate_preconditioner(
    cfg: PreconditionerConfig,
    *,
    service: PreconditionerService,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    matrix_path: Path,
    matrix_index: int,
    params: SolverParams,
    display_name: str | None = None,
    x_exact: torch.Tensor | None = None,
    settings: NeurallsSettings,
    cache: SystemPreconditionerCache | None = None,
    notation_policy: NotationPolicy | None = None,
) -> PreconditionerComparisonEntry:
    """Build one preconditioner, run it, and package its evaluation outcome.

    The constructed preconditioner (and, for neural preconditioners, its loaded
    checkpoint) is only referenced by locals in this call — it goes out of scope
    on return, so the caller's loop never holds more than one model at a time.
    Exception: a preconditioner `cache` stores reuses it across sibling calls
    (see `SystemPreconditionerCache`), by design — the caller owns that
    cache's lifetime, scoped to the whole sweep group, not to this call.

    Args:
        cfg: Preconditioner configuration to build and evaluate.
        service: Shared preconditioner factory service.
        matrix: System matrix (shape: n x n).
        rhs: Right-hand side vector (shape: n,).
        matrix_path: Dataset directory for extra-input binding (may not be a directory).
        matrix_index: Sample index for extra-input extraction.
        params: Solver tolerances and iteration limits.
        display_name: Optional human-readable comparison label, for logging.
        x_exact: Reference solution shared across every preconditioner in this
            comparison (see ``compare_preconditioners``), or ``None``.
        settings: Resolved runtime roots, used to resolve any dataset
            directory feeding this preconditioner's generation cost.
        cache: Optional per-sweep-group cache letting a preconditioner built
            for an earlier sibling system be reused here instead of rebuilt,
            when doing so is provably valid (see `SystemPreconditionerCache`).
            `None` (the default) means every call builds fresh, unchanged
            from before this parameter existed.
        notation_policy: Render target and symbol overrides for the display name
            (from ``[comparison_baseline.presentation]``); ``None`` is plain text.

    Returns:
        Named result containing the solve outcome and plot metadata for ``cfg``.
    """
    logger.info("Evaluating preconditioner: {}", cfg.name)
    color_key, marker_key = _pod2g_style_keys(cfg)
    try:
        return _run_preconditioner(
            cfg,
            service=service,
            matrix=matrix,
            rhs=rhs,
            matrix_path=matrix_path,
            matrix_index=matrix_index,
            params=params,
            color_key=color_key,
            marker_key=marker_key,
            x_exact=x_exact,
            settings=settings,
            cache=cache,
            notation_policy=notation_policy,
        )
    except Exception as exc:  # noqa: BLE001
        # Broad by design: one preconditioner's failure (build, solve, CUDA OOM,
        # ...) must never abort the rest of the comparison.
        logger.opt(exception=True).warning(
            "Preconditioner '{}' failed (comparison={}): {}: {}",
            cfg.name,
            display_name or "unnamed",
            type(exc).__name__,
            exc,
        )
        release_device_memory()
        return PreconditionerComparisonEntry(
            name=cfg.name,
            result=_breakdown_result(cfg.name, rhs=rhs, error=f"{type(exc).__name__}: {exc}"),
            label=cfg.name,
            family=preconditioner_family(cfg),
            color_key=color_key,
            marker_key=marker_key,
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


def _run_preconditioner(
    cfg: PreconditionerConfig,
    *,
    service: PreconditionerService,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    matrix_path: Path,
    matrix_index: int,
    params: SolverParams,
    color_key: str | None,
    marker_key: str | None,
    x_exact: torch.Tensor | None = None,
    settings: NeurallsSettings,
    cache: SystemPreconditionerCache | None = None,
    notation_policy: NotationPolicy | None = None,
) -> PreconditionerComparisonEntry:
    """Build and solve one preconditioner; raises on any failure.

    ``setup_usage`` wraps both the preconditioner's construction+``.setup()``
    (``PreconditionerService.create_preconditioner`` — every `builders.py`
    builder calls `.setup(matrix)` before returning) and any scheduling
    wrapper's own `.setup()` (`_create_scheduled_preconditioners`). Every
    `torchalg.Preconditioner` requires an explicit `setup(matrix)` call
    before `apply()`, so this measurement is always the real build cost for
    every preconditioner kind (IC0/Jacobi's cheap factorization, AMG/POD-2G's
    hierarchy build, AdaptiveSA/BootstrapAMG's bootstrap cycles) — never a
    near-zero artifact of timing construction alone while the real work
    happened lazily inside `apply()`.

    Checkpoint loading is deliberately *outside* the timed region, for every
    checkpoint-backed kind alike (neural, neural AMG, fitted POD-2G basis,
    neural POD-2G's predictor), and so outside `setup_peak_memory_bytes`
    too: it is file I/O that a deployment pays once, not algorithmic setup
    cost. `PreconditionerService.preload` loads each checkpoint once per
    method (the store lives as long as the method workload), and the timed
    build only borrows the loaded object. A neural preconditioner's setup is
    matrix-independent, so its setup cost is therefore near zero by design; the
    checkpoint and data loading are excluded from the benchmark cost model. Download and hashing are
    outside the measurement as well (they run in the prepare phase, before any
    MLflow run opens). A service built without a store loads inside the build,
    and then inside the timer.

    On a `cache` hit, none of that rebuilding happens at all: the already-
    ready object from an earlier sibling system is reused directly, and the
    *real* measured cost from that earlier build is reported again rather
    than a fresh (near-zero, misleadingly "free") measurement — the whole
    point of caching is that this cost is genuinely paid once, so every
    sibling's result should keep showing its true size.
    """
    family = preconditioner_family(cfg)
    staged_inputs = _load_extra_inputs(matrix_path, matrix_index)
    cached = cache.get(cfg.name) if cache is not None else None
    if cached is not None:
        scheduled: dict[str, Preconditioner] = {cfg.name: cached.preconditioner}
        setup_cost_seconds = cached.setup_cost_seconds
        setup_peak_memory_bytes = cached.setup_peak_memory_bytes
        setup_time_samples = ()
    else:
        service.preload(matrix, cfg)
        timed_setup = repeat_timed(
            lambda: _build_scheduled_and_bind(service, matrix, cfg, staged_inputs),
            params.timing,
            device=matrix.device,
        )
        scheduled = timed_setup.value
        setup_cost_seconds = timed_setup.seconds
        setup_peak_memory_bytes = timed_setup.peak_memory_bytes
        setup_time_samples = timed_setup.samples_seconds
        if cache is not None:
            cache.put(
                cfg.name,
                scheduled[cfg.name],
                setup_cost_seconds=setup_cost_seconds,
                setup_peak_memory_bytes=setup_peak_memory_bytes,
            )
    if cached is not None:
        _bind_extra_inputs(scheduled, matrix, staged_inputs)
    base_label = build_preconditioner_labels(scheduled, {cfg.name: cfg}, policy=notation_policy)[
        cfg.name
    ]
    dataset_label = Path(color_key).name if color_key is not None else None
    label = contextualize_preconditioner_label(base_label, dataset_label)
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
    )[cfg.name]
    generation_cost = resolve_generation_cost(cfg, settings=settings)
    result = replace(
        result,
        generation_cost=generation_cost,
        setup_cost=setup_cost_seconds,
        setup_peak_memory_bytes=setup_peak_memory_bytes,
        setup_time_samples=setup_time_samples,
    )
    return PreconditionerComparisonEntry(
        name=cfg.name,
        result=result,
        label=label,
        family=family,
        color_key=color_key,
        marker_key=marker_key,
    )


def _execute_matrix_bundle(
    method: ResolvedMethod,
    bundle: MatrixWorkload,
    *,
    service: PreconditionerService,
    params: SolverParams,
    device: torch.device,
    offline_cost: StageCost | None,
    observer: WorkloadMemoryObserver,
    setup_events: list[MatrixSetupEvent],
    attempts: list[SolveAttempt],
    diagnostics: list[tuple[SystemId, CGComparisonResult]],
    realizations: list[SystemRealization],
) -> None:
    """Execute one matrix and release all device state on every exit path.

    Realized facts are read from the built preconditioner right after its one setup and
    appended to ``realizations``; only those plain facts outlive the release below.
    """
    matrix: torch.Tensor | None = None
    staged_inputs: tuple[np.ndarray, ...] = ()
    scheduled: dict[str, Preconditioner] | None = None
    try:
        matrix = torch.as_tensor(bundle.matrix).to(device)
        staged_inputs = _load_extra_inputs(bundle.matrix_path, bundle.matrix_index)
        event_id = SetupEventId(f"{method.name}:{bundle.dataset}:{int(bundle.matrix_id)}")
        try:
            service.preload(matrix, method.config)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Recoverable setup failure for method '{}' on dataset '{}' matrix {}: {}: {}",
                method.name,
                bundle.dataset,
                int(bundle.matrix_id),
                type(exc).__name__,
                exc,
            )
            setup_events.append(
                MatrixSetupEvent(
                    event_id=event_id,
                    dataset=bundle.dataset,
                    matrix_id=bundle.matrix_id,
                    method=method.name,
                    setup_s=0.0,
                    samples_s=(),
                    peak_memory=MeasuredPeakMemory(0),
                    outcome=SetupFailed(f"{type(exc).__name__}: {exc}"),
                )
            )
            diagnostics.extend(
                (
                    member.tag.system_id,
                    _breakdown_result(
                        method.name,
                        rhs=torch.as_tensor(member.rhs),
                        error=f"setup failed: {exc}",
                    ),
                )
                for member in bundle.members
            )
            return
        timed_setup = repeat_timed_outcome(
            lambda matrix=matrix, staged_inputs=staged_inputs: _build_scheduled_and_bind(
                service, matrix, method.config, staged_inputs
            ),
            params.timing,
            device=device,
        )
        observer.observe_operation(timed_setup.peak_memory_bytes)
        if isinstance(timed_setup, FailedTimedRuns):
            logger.warning(
                "Recoverable setup failure for method '{}' on dataset '{}' matrix {}: {}: {}",
                method.name,
                bundle.dataset,
                int(bundle.matrix_id),
                type(timed_setup.error).__name__,
                timed_setup.error,
            )
            setup_events.append(
                MatrixSetupEvent(
                    event_id=event_id,
                    dataset=bundle.dataset,
                    matrix_id=bundle.matrix_id,
                    method=method.name,
                    setup_s=timed_setup.seconds,
                    samples_s=timed_setup.samples_seconds,
                    peak_memory=MeasuredPeakMemory(timed_setup.peak_memory_bytes),
                    outcome=SetupFailed(f"{type(timed_setup.error).__name__}: {timed_setup.error}"),
                )
            )
            diagnostics.extend(
                (
                    member.tag.system_id,
                    _breakdown_result(
                        method.name,
                        rhs=torch.as_tensor(member.rhs),
                        error=f"setup failed: {timed_setup.error}",
                    ),
                )
                for member in bundle.members
            )
            return

        scheduled = timed_setup.value
        realizations.extend(capture_system_realizations(scheduled[method.name], bundle))
        setup_events.append(
            MatrixSetupEvent(
                event_id=event_id,
                dataset=bundle.dataset,
                matrix_id=bundle.matrix_id,
                method=method.name,
                setup_s=timed_setup.seconds,
                samples_s=timed_setup.samples_seconds,
                peak_memory=MeasuredPeakMemory(timed_setup.peak_memory_bytes),
                outcome=SetupCompleted(),
            )
        )

        for member in bundle.members:
            rhs = torch.as_tensor(member.rhs).to(device=device, dtype=matrix.dtype)
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
            if offline_cost is not None:
                result = replace(result, generation_cost=offline_cost)
            diagnostics.append((member.tag.system_id, result))
            solve_peak = result.solve_peak_memory_bytes
            if solve_peak is not None:
                observer.observe_operation(solve_peak)
            outcome = (
                SolveConverged(result.iterations, result.residual)
                if result.converged
                else SolveFailed(result.iterations, result.error or "solver did not converge")
            )
            if isinstance(outcome, SolveFailed):
                logger.warning(
                    "Recoverable solve failure for method '{}' on system '{}' "
                    "(dataset '{}', matrix {}, RHS {}): {}",
                    method.name,
                    member.tag.system_id,
                    bundle.dataset,
                    int(bundle.matrix_id),
                    member.tag.rhs_ordinal,
                    outcome.reason,
                )
            attempts.append(
                SolveAttempt(
                    setup_event_id=event_id,
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
    finally:
        if scheduled is not None:
            del scheduled
        if matrix is not None:
            del matrix
        del staged_inputs
        release_device_memory()


def _executed_presentation(
    prepared: PreparedMethodPresentation, realizations: tuple[SystemRealization, ...]
) -> ExecutedMethodPresentation:
    """Upgrade to the realized state only when some setup produced facts to retain."""
    return realize_method(prepared, realizations) if realizations else prepared


def execute_method_workload(
    method: ResolvedMethod,
    bundles: Iterable[MatrixWorkload],
    *,
    prepared: PreparedMethodPresentation,
    params: SolverParams,
    settings: NeurallsSettings,
    notation_policy: NotationPolicy,
) -> CompletedMethodWorkload:
    """Execute one method over matrix bundles, keeping matrix state RHS-local.

    Dataset/checkpoint reads, reference solves, and reporting are outside the
    setup/solve timers.  Each matrix receives one fresh prepared method object;
    sibling RHS solves share that exact object.  The method-owned checkpoint
    store spans matrices and is released before this function returns.

    The workload reports from ``prepared`` until a setup succeeds; facts read from the
    built preconditioner then upgrade it to the realized state. ``prepared`` must belong
    to ``method``.
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

    with CheckpointStore() as checkpoints:
        service = PreconditionerService(checkpoints=checkpoints)
        for bundle in bundles:
            planned_list.extend(member.tag for member in bundle.members)
            _execute_matrix_bundle(
                method,
                bundle,
                service=service,
                params=params,
                device=device,
                offline_cost=offline_cost,
                observer=observer,
                setup_events=setup_events,
                attempts=attempts,
                diagnostics=diagnostics,
                realizations=realizations,
            )
            del bundle

    color_key, marker_key = _pod2g_style_keys(method.config)
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
        presentation=_executed_presentation(prepared, tuple(realizations)),
        plot_styles={
            str(method.name): PreconditionerPlotStyle(
                label=contextualize_preconditioner_label(
                    build_configured_preconditioner_label(method.config, policy=notation_policy),
                    Path(color_key).name if color_key is not None else None,
                ),
                family=preconditioner_family(method.config),
                color_key=color_key,
                marker_key=marker_key,
            )
        },
    )


def build_comparison_result_from_diagnostics(
    *,
    general_params: ComparisonGeneral,
    preconditioner_configs: Sequence[PreconditionerConfig],
    resolved_input: ResolvedComparisonInput,
    results: dict[str, CGComparisonResult],
    output_root: Path,
    display_name: str | None,
    notation_policy: NotationPolicy,
) -> ComparisonResult:
    """Build the retained per-system result/artifact payload without re-solving.

    Method-major execution owns solver work. This adapter recreates only the
    public diagnostic envelope (plots, shapes, labels, and summaries) expected
    by existing nested comparison runs and artifact consumers.
    """
    paths = _resolve_comparison_paths(
        general_params=general_params, output_root=output_root, figures_root=None
    )
    _ensure_comparison_directories(paths)
    context = build_comparison_source_context(paths, resolved_input)
    plot_styles = {}
    for config in preconditioner_configs:
        if config.name not in results:
            continue
        color_key, marker_key = _pod2g_style_keys(config)
        dataset_label = Path(color_key).name if color_key is not None else None
        plot_styles[config.name] = PreconditionerPlotStyle(
            label=contextualize_preconditioner_label(
                build_configured_preconditioner_label(config, policy=notation_policy),
                dataset_label,
            ),
            family=preconditioner_family(config),
            color_key=color_key,
            marker_key=marker_key,
        )
    styles = plot_style_maps(plot_styles)
    plot_paths = _generate_comparison_plots(
        results,
        paths,
        styles.labels,
        comparison_context=context,
        families=styles.families,
        color_keys=styles.color_keys,
        marker_keys=styles.marker_keys,
        system_size=int(resolved_input.matrix.shape[0]),
        rtol=general_params.params.rtol,
        atol=general_params.params.atol,
        max_iterations=general_params.params.max_iterations,
    )
    del display_name  # kept in the boundary for parity with compare_preconditioners
    return ComparisonResult(
        results=results,
        summary=format_results_summary(results),
        plot_paths=plot_paths,
        preconditioners=tuple(config.name for config in preconditioner_configs),
        solver_params=general_params,
        recommendations=summarize_best_combinations(results),
        output_dir=paths.output,
        matrix_shape=tuple(resolved_input.matrix.shape),
        rhs_shape=tuple(resolved_input.rhs.shape),
        rhs_source_kind=resolved_input.rhs_source_kind,
        plot_styles=plot_styles,
    )


def _resolve_comparison_paths(
    *,
    general_params: ComparisonGeneral,
    output_root: Path | None,
    figures_root: Path | None,
) -> ComparisonPaths:
    """Resolve all paths for a comparison run.

    Args:
        general_params: Comparison general configuration with params+data context.
        output_root: Optional override for output root directory.
        figures_root: Optional override for figures directory.

    Returns:
        ComparisonPaths with all resolved and validated paths.

    Raises:
        ValueError: If matrix_path or rhs_path are missing.
    """
    matrix_file = Path(general_params.data.matrix_path)
    rhs_file = (
        Path(general_params.data.rhs_path)
        if general_params.data.rhs_path is not None
        else matrix_file
    )

    if output_root is not None:
        output_base = resolve_user_path(output_root)
    else:
        output_base = (Path.cwd() / "comparison" / matrix_file.stem).resolve()

    figs_base = Path(figures_root) if figures_root else output_base / "figures"

    return ComparisonPaths(
        matrix=matrix_file,
        rhs=rhs_file,
        output=output_base,
        figures=figs_base,
    )


def _ensure_comparison_directories(paths: ComparisonPaths) -> None:
    """Ensure the figures directory exists.

    Only the figures subdirectory needs explicit creation; the output root
    is either caller-supplied or an MLflow-managed artifact directory.
    mkdir(parents=True) inside ensure_dir creates any intermediate directories
    (including paths.output) as a side effect.

    Args:
        paths: Comparison paths with output and figures directories.
    """
    ensure_dir(paths.figures)


def compare_preconditioners(
    *,
    general_params: ComparisonGeneral,
    preconditioner_configs: Sequence[PreconditionerConfig],
    output_root: Path | None = None,
    figures_root: Path | None = None,
    display_name: str | None = None,
    resolved_input: ResolvedComparisonInput | None = None,
    evaluation_mapper: PreconditionerEvaluationMapper = map,
    settings: NeurallsSettings,
    cache: SystemPreconditionerCache | None = None,
    notation_policy: NotationPolicy | None = None,
) -> ComparisonResult:
    """Run CG comparisons and generate diagnostics.

    Orchestrates a 6-step workflow:
    1. Validate inputs
    2. Resolve paths (matrix, rhs, output, figures)
    3. Load, validate, and place the linear system on the solver device
    4. Evaluate each preconditioner config: build, bind, and solve
       (one preconditioner resident at a time — see ``evaluation_mapper``)
    5. Generate diagnostic plots
    6. Package and return result

    Step 4 evaluates preconditioners one at a time rather than building every
    config's preconditioner up front: each config's model (including any
    GPU-resident neural checkpoint) is constructed, used, and released before
    the next config is built, bounding peak memory to a single model.

    Args:
        general_params: Comparison general configuration.
        preconditioner_configs: Sequence of preconditioner configurations.
        output_root: Optional override for output root directory.
        figures_root: Optional override for figures directory.
        display_name: Optional human-readable label included in failure logs.
        evaluation_mapper: Strategy for running the per-config evaluations in
            ``preconditioner_configs``. Defaults to the builtin ``map``, i.e.
            sequential execution with exactly one preconditioner resident at a
            time. Matches the call signature of ``concurrent.futures.Executor.map``,
            so passing e.g. ``ThreadPoolExecutor(max_workers=n).map`` runs up to
            ``n`` models concurrently without any other change here — an opt-in
            left to the caller, who is best placed to judge whether ``n`` models'
            worth of GPU memory fit at once.
        settings: Resolved runtime roots, used to resolve any dataset
            directory feeding a preconditioner's generation cost — see
            ``_generation_cost.py::resolve_generation_cost``.
        cache: Optional per-sweep-group cache letting each preconditioner
            built here be reused by a later sibling system's call to this
            same function, when doing so is provably valid (see
            ``SystemPreconditionerCache``). ``None`` (the default) means
            every call builds every preconditioner fresh, unchanged from
            before this parameter existed — only a sweep-aware caller
            passes a real cache, scoped to its own sweep group. A cache also
            carries its method-workload checkpoint store, so checkpoints are loaded
            once per method workload; without a cache (or a store on it) this call uses
            a private store released when it returns, so the load is still
            outside the timer.
        notation_policy: Render target and symbol overrides for preconditioner
            display names, from ``[comparison_baseline.presentation]``. ``None``
            (the default) renders plain text with the default symbols.

    Returns:
        ComparisonResult with results, summary, plot paths, and solver metadata.

    Raises:
        ValueError: If no preconditioner configs provided or required paths missing.
        FileNotFoundError: If matrix/rhs files don't exist.
    """
    if not preconditioner_configs:
        raise ValueError("At least one preconditioner config must be provided.")
    expected_keys = validate_unique_preconditioner_keys(preconditioner_configs)

    paths = _resolve_comparison_paths(
        general_params=general_params,
        output_root=output_root,
        figures_root=figures_root,
    )
    _ensure_comparison_directories(paths)
    solver_device = resolve_device()
    comparison_context = build_comparison_source_context(paths, resolved_input)
    logger.info("Comparison: {} | device={}", comparison_context, solver_device)

    resolved_matrix_index = general_params.data.matrix_index

    system = _load_linear_system(
        paths,
        rhs_sample_index=0,
        matrix_index=resolved_matrix_index,
        normalize_system=general_params.data.normalize_system,
        resolved_input=resolved_input,
    )
    matrix = system.matrix.to(solver_device)
    rhs = system.rhs.to(solver_device)
    # One A/b pair has one true solution: compute it once on the selected
    # solver device and share it across every preconditioner below instead of
    # each one redoing this Jacobi-PCG solve from scratch.
    x_exact = compute_reference_solution(
        matrix,
        rhs,
        rtol=general_params.params.rtol,
        margin=general_params.params.reference_precision_margin,
    )

    results: dict[str, CGComparisonResult] = {}
    plot_styles: dict[str, PreconditionerPlotStyle] = {}
    with contextlib.ExitStack() as stack:
        shared_store = cache.checkpoints if cache is not None else None
        store = shared_store if shared_store is not None else stack.enter_context(CheckpointStore())
        service = PreconditionerService(checkpoints=store)
        evaluate_one = partial(
            _evaluate_preconditioner,
            service=service,
            matrix=matrix,
            rhs=rhs,
            matrix_path=paths.matrix,
            matrix_index=resolved_matrix_index,
            params=general_params.params,
            display_name=display_name,
            x_exact=x_exact,
            settings=settings,
            cache=cache,
            notation_policy=notation_policy,
        )

        for entry in evaluation_mapper(evaluate_one, preconditioner_configs):
            if entry.name not in expected_keys:
                raise RuntimeError(
                    f"Evaluation returned unexpected preconditioner key {entry.name!r}; "
                    f"expected one of {expected_keys!r}."
                )
            if entry.name in results:
                raise RuntimeError(
                    f"Evaluation returned preconditioner key {entry.name!r} more than once."
                )
            if entry.result.preconditioner != entry.name:
                raise RuntimeError(
                    f"Evaluation result for {entry.name!r} identifies itself as "
                    f"{entry.result.preconditioner!r}."
                )
            results[entry.name] = entry.result
            plot_styles[entry.name] = PreconditionerPlotStyle(
                label=entry.label,
                family=entry.family,
                color_key=entry.color_key,
                marker_key=entry.marker_key,
            )

        missing_keys = tuple(key for key in expected_keys if key not in results)
        if missing_keys:
            raise RuntimeError(
                f"Evaluation returned no result for preconditioner keys {missing_keys!r}."
            )

    recommendations = ComparisonRecommendations()

    style = plot_style_maps(plot_styles)
    plot_paths = _generate_comparison_plots(
        results,
        paths,
        style.labels,
        comparison_context=comparison_context,
        families=style.families,
        color_keys=style.color_keys,
        marker_keys=style.marker_keys,
        system_size=int(matrix.shape[0]),
        rtol=general_params.params.rtol,
        atol=general_params.params.atol,
        max_iterations=general_params.params.max_iterations,
    )

    return ComparisonResult(
        results=results,
        summary=format_results_summary(results),
        plot_paths=plot_paths,
        preconditioners=tuple(cfg.name for cfg in preconditioner_configs),
        solver_params=general_params,
        recommendations=recommendations,
        output_dir=paths.output,
        matrix_shape=(matrix.shape[0], matrix.shape[1]),
        rhs_shape=tuple(rhs.shape),
        rhs_source_kind=resolved_input.rhs_source_kind if resolved_input is not None else None,
        plot_styles=plot_styles,
    )
