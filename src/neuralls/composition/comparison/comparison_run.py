"""One comparison run across all configured preconditioners — pure orchestration entry point."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torchalg.preconditioners.base import Preconditioner
from torchalg.utils.device import resolve_device

from neuralls.composition.comparison._generation_cost import resolve_generation_cost
from neuralls.composition.comparison._plots import _generate_comparison_plots
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    _bind_extra_inputs,
    _create_scheduled_preconditioners,
    _load_extra_inputs,
    capture_system_realizations,
)
from neuralls.composition.comparison.models import (
    ComparisonPaths,
    ComparisonPresentation,
    CompletedMethodWorkload,
    MatrixWorkload,
    ResolvedComparisonInput,
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
from neuralls.platform.reporting.figure_text import ComparisonFigureTitle, Legend, method_legend
from neuralls.platform.storage.filesystem import ensure_dir
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
    method_text: str,
    dataset_text: str,
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
                "Recoverable setup failure for {} on {}, matrix {}: {}: {}",
                method_text,
                dataset_text,
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
                "Recoverable setup failure for {} on {}, matrix {}: {}: {}",
                method_text,
                dataset_text,
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
        realizations.extend(
            capture_system_realizations(
                scheduled[method.name], PreconditionerName(method.name), bundle
            )
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
                    "Recoverable solve failure for {} on {}, matrix {}, RHS {}: {}",
                    method_text,
                    dataset_text,
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
        service = PreconditionerService(FS_FIT_DATA, checkpoints=checkpoints)
        for bundle in bundles:
            planned_list.extend(member.tag for member in bundle.members)
            _execute_matrix_bundle(
                method,
                bundle,
                method_text=method_text,
                dataset_text=render_dataset_label(datasets[bundle.dataset], notation),
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


def build_comparison_result_from_diagnostics(
    *,
    general_params: ComparisonGeneral,
    preconditioner_configs: Sequence[PreconditionerConfig],
    resolved_input: ResolvedComparisonInput,
    results: dict[str, CGComparisonResult],
    output_root: Path,
    presentation: ComparisonPresentation,
) -> ComparisonResult:
    """Build the retained per-system result/artifact payload without re-solving.

    Method-major execution owns solver work. This adapter recreates only the
    public diagnostic envelope (plots, shapes, legends, and summaries) expected
    by existing nested comparison runs and artifact consumers; every method is
    presented from ``presentation``, the cohort prepared at preflight.
    """
    paths = _resolve_comparison_paths(
        general_params=general_params, output_root=output_root, figures_root=None
    )
    _ensure_comparison_directories(paths)
    plot_styles = {}
    for config in preconditioner_configs:
        if config.name not in results:
            continue
        color_key, marker_key = _pod2g_style_keys(config)
        plot_styles[config.name] = PreconditionerPlotStyle(
            presentation=presentation.methods.method(PreconditionerName(config.name)),
            family=preconditioner_family(config),
            color_key=color_key,
            marker_key=marker_key,
        )
    styles = plot_style_maps(plot_styles)
    plot_paths = _generate_comparison_plots(
        results,
        paths,
        _legend_of(plot_styles, presentation),
        ComparisonFigureTitle(
            presentation.title, int(resolved_input.matrix.shape[0]), presentation.notation
        ),
        families=styles.families,
        color_keys=styles.color_keys,
        marker_keys=styles.marker_keys,
        rtol=general_params.params.rtol,
        atol=general_params.params.atol,
        max_iterations=general_params.params.max_iterations,
    )
    return ComparisonResult(
        results=results,
        summary=format_results_summary(
            results,
            {name: style.presentation.compact for name, style in plot_styles.items()},
            presentation.notation,
        ),
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


def _legend_of(
    plot_styles: Mapping[str, PreconditionerPlotStyle], presentation: ComparisonPresentation
) -> Legend:
    """The legend of a comparison's methods, each rendered from its own prepared presentation."""
    return method_legend(
        {name: style.presentation.compact for name, style in plot_styles.items()},
        presentation.notation,
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
