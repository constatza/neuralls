"""One comparison run across all configured preconditioners — pure orchestration entry point."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path

import torch
from loguru import logger
from torchalg.utils.device import resolve_device

from neuralls.composition.comparison._linear_system import (
    _load_linear_system,
    _log_matrix_condition_number,
)
from neuralls.composition.comparison._plots import _generate_comparison_plots
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    _create_scheduled_preconditioners,
    _load_and_bind_extra_inputs,
)
from neuralls.composition.comparison.models import (
    ComparisonPaths,
    PreconditionerComparisonEntry,
    ResolvedComparisonInput,
)
from neuralls.domain.analysis.spectra import PreconditionerCallable, compute_condition_numbers
from neuralls.domain.solver.comparison import (
    _to_numpy,
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
from neuralls.platform.config.models.preconditioner import PreconditionerConfig, PreconditionerType
from neuralls.platform.config.models.preconditioner_family import (
    PreconditionerFamilyKey,
    preconditioner_family,
)
from neuralls.platform.config.resolution import resolve_user_path
from neuralls.platform.reporting.preconditioner_labels import build_preconditioner_labels
from neuralls.platform.storage.filesystem import ensure_dir

type PreconditionerEvaluationMapper = Callable[
    [
        Callable[[PreconditionerConfig], PreconditionerComparisonEntry],
        Sequence[PreconditionerConfig],
    ],
    Iterable[PreconditionerComparisonEntry],
]


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

    Returns:
        Named result: solve outcome, condition number, and plot label for ``cfg``.
    """
    logger.info(f"Preconditioner: {cfg.name} (comparison={display_name or 'unnamed'})")
    base_preconditioners = {cfg.name: service.create_preconditioner(matrix, cfg)}
    scheduled = _create_scheduled_preconditioners(
        preconditioner_configs=[cfg],
        matrix=matrix,
        base_preconditioners=base_preconditioners,
    )
    _load_and_bind_extra_inputs(
        scheduled, matrix=matrix, matrix_path=matrix_path, matrix_index=matrix_index
    )

    cond_callables: dict[str, PreconditionerCallable] = {name: p for name, p in scheduled.items()}
    condition_number = compute_condition_numbers(_to_numpy(matrix), cond_callables)[cfg.name]
    label = build_preconditioner_labels(scheduled)[cfg.name]
    result = run_cg_comparison(
        matrix,
        rhs,
        preconditioners=scheduled,
        rtol=params.rtol,
        atol=params.atol,
        maxiter=params.max_iterations,
        m_max=params.m_max,
    )[cfg.name]
    return PreconditionerComparisonEntry(
        name=cfg.name,
        result=result,
        condition_number=condition_number,
        label=label,
        family=preconditioner_family(cfg),
    )


def _log_solver_device(display_name: str | None) -> None:
    """Log which torch device the CG solver used for this comparison run.

    Args:
        display_name: Optional human-readable comparison label.
    """
    device = resolve_device()
    logger.info(f"CG solver device: comparison={display_name or 'unnamed'} device={device}")


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
    3. Load and validate linear system
    4. Evaluate each preconditioner config: build, bind, compute condition number, solve
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
        display_name: Optional display name for plots.
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
    _log_solver_device(display_name)

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
    condition_number_raw = _log_matrix_condition_number(
        _to_numpy(system.matrix),
        matrix_path=paths.matrix,
        display_name=display_name,
    )

    service = PreconditionerService()
    evaluate_one = partial(
        _evaluate_preconditioner,
        service=service,
        matrix=system.matrix,
        rhs=system.rhs,
        matrix_path=paths.matrix,
        matrix_index=resolved_matrix_index,
        params=general_params.params,
        display_name=display_name,
    )

    results: dict[str, CGComparisonResult] = {}
    cond_numbers: dict[str, float] = {}
    labels: dict[str, str] = {}
    families: dict[str, PreconditionerFamilyKey] = {}
    for entry in evaluation_mapper(evaluate_one, preconditioner_configs):
        results[entry.name] = entry.result
        cond_numbers[entry.name] = entry.condition_number
        labels[entry.name] = entry.label
        families[entry.name] = entry.family

    if "none" not in results:
        baseline = run_cg_comparison(
            system.matrix,
            system.rhs,
            preconditioners={},
            rtol=general_params.params.rtol,
            atol=general_params.params.atol,
            maxiter=general_params.params.max_iterations,
            m_max=general_params.params.m_max,
        )
        results["none"] = baseline["none"]
        families.setdefault("none", PreconditionerType.NONE)

    recommendations = ComparisonRecommendations()

    plot_paths = _generate_comparison_plots(
        results,
        cond_numbers,
        paths,
        labels,
        families,
        display_name=display_name,
        rtol=general_params.params.rtol,
        atol=general_params.params.atol,
        max_iterations=general_params.params.max_iterations,
    )

    return ComparisonResult(
        results=results,
        summary=format_results_summary(results),
        plot_paths=plot_paths,
        preconditioners=tuple(cfg.name for cfg in preconditioner_configs),
        condition_numbers=cond_numbers,
        solver_params=general_params,
        recommendations=recommendations,
        output_dir=paths.output,
        matrix_shape=(system.matrix.shape[0], system.matrix.shape[1]),
        rhs_shape=tuple(system.rhs.shape),
        condition_number_raw=condition_number_raw,
        rhs_source_kind=resolved_input.rhs_source_kind if resolved_input is not None else None,
    )
