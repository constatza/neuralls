"""One comparison run across all configured preconditioners — pure orchestration entry point."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path

import torch
from loguru import logger
from torchalg.utils.device import resolve_device

from neuralls.composition.comparison._linear_system import _load_linear_system
from neuralls.composition.comparison._plots import _generate_comparison_plots
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    _create_scheduled_preconditioners,
    _load_and_bind_extra_inputs,
)
from neuralls.composition.comparison._presentation import build_comparison_source_context
from neuralls.composition.comparison.models import (
    ComparisonPaths,
    PreconditionerComparisonEntry,
    ResolvedComparisonInput,
)
from neuralls.domain.solver.comparison import (
    _to_numpy,
    compute_reference_solution,
    format_results_summary,
    run_cg_comparison,
)
from neuralls.domain.solver.models.config import SolverParams
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonRecommendations,
    ComparisonResult,
)
from neuralls.platform.config.models.comparison import ComparisonGeneral
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
    PreconditionerType,
)
from neuralls.platform.config.models.preconditioner_family import (
    PreconditionerFamilyKey,
    preconditioner_family,
)
from neuralls.platform.config.resolution import resolve_user_path
from neuralls.platform.reporting.preconditioner_labels import build_preconditioner_labels
from neuralls.platform.storage.filesystem import ensure_dir
from neuralls.shared.device import release_device_memory

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
) -> PreconditionerComparisonEntry:
    """Build one preconditioner, run it, and package its evaluation outcome.

    The constructed preconditioner (and, for neural preconditioners, its loaded
    checkpoint) is only referenced by locals in this call — it goes out of scope
    on return, so the caller's loop never holds more than one model at a time.

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
) -> PreconditionerComparisonEntry:
    """Build and solve one preconditioner; raises on any failure."""
    family = preconditioner_family(cfg)
    base_preconditioners = {cfg.name: service.create_preconditioner(matrix, cfg)}
    scheduled = _create_scheduled_preconditioners(
        preconditioner_configs=[cfg],
        matrix=matrix,
        base_preconditioners=base_preconditioners,
    )
    _load_and_bind_extra_inputs(
        scheduled, matrix=matrix, matrix_path=matrix_path, matrix_index=matrix_index
    )
    label = build_preconditioner_labels(scheduled, {cfg.name: family})[cfg.name]
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
    )[cfg.name]
    return PreconditionerComparisonEntry(
        name=cfg.name,
        result=result,
        label=label,
        family=family,
        color_key=color_key,
        marker_key=marker_key,
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
) -> ComparisonResult:
    """Run CG comparisons and generate diagnostics.

    Orchestrates a 7-step workflow:
    1. Validate inputs
    2. Resolve paths (matrix, rhs, output, figures)
    3. Load, validate, and place the linear system on the solver device
    4. Evaluate each preconditioner config: build, bind, and solve
       (one preconditioner resident at a time — see ``evaluation_mapper``)
    5. Add the "none" (identity) baseline if no config produced one
    6. Generate diagnostic plots
    7. Package and return result

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

    Returns:
        ComparisonResult with results, summary, plot paths, and solver metadata.

    Raises:
        ValueError: If no preconditioner configs provided or required paths missing.
        FileNotFoundError: If matrix/rhs files don't exist.
    """
    if not preconditioner_configs:
        raise ValueError("At least one preconditioner config must be provided.")

    paths = _resolve_comparison_paths(
        general_params=general_params,
        output_root=output_root,
        figures_root=figures_root,
    )
    _ensure_comparison_directories(paths)
    solver_device = resolve_device()
    comparison_context = build_comparison_source_context(paths, resolved_input)
    logger.info("Comparison: {} | device={}", comparison_context, solver_device)

    resolved_matrix_index = (
        general_params.data.matrix_index if general_params.data.matrix_index is not None else 0
    )

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

    service = PreconditionerService()
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
    )

    results: dict[str, CGComparisonResult] = {}
    labels: dict[str, str] = {}
    families: dict[str, PreconditionerFamilyKey] = {}
    color_keys: dict[str, str] = {}
    marker_keys: dict[str, str] = {}
    for entry in evaluation_mapper(evaluate_one, preconditioner_configs):
        results[entry.name] = entry.result
        labels[entry.name] = entry.label
        families[entry.name] = entry.family
        if entry.color_key is not None:
            color_keys[entry.name] = entry.color_key
        if entry.marker_key is not None:
            marker_keys[entry.name] = entry.marker_key

    if "none" not in results:
        baseline = run_cg_comparison(
            matrix,
            rhs,
            preconditioners={},
            x_exact=x_exact,
            rtol=general_params.params.rtol,
            atol=general_params.params.atol,
            maxiter=general_params.params.max_iterations,
            m_max=general_params.params.m_max,
            reference_precision_margin=general_params.params.reference_precision_margin,
        )
        results["none"] = baseline["none"]
        families.setdefault("none", PreconditionerType.NONE)

    recommendations = ComparisonRecommendations()

    plot_paths = _generate_comparison_plots(
        results,
        paths,
        labels,
        comparison_context=comparison_context,
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
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
    )
