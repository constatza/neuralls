"""RHS and solution archive path resolution for the generate workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from neuralls.composition.generation._context_builder import DataGenerationContext
from neuralls.domain.generation.plan import StrategySpec
from neuralls.domain.generation.runner import run_generation
from neuralls.platform.config.models.data_models import GenerationConfig


def _merge_rhs_archive_options(
    strategy: StrategySpec | None,
    generation_cfg: GenerationConfig,
) -> dict[str, Any]:
    """Combine generation-level CG options with strategy-specific overrides.

    Args:
        strategy: Optional strategy spec whose options take precedence.
        generation_cfg: Generation config providing generation-level defaults.

    Returns:
        Merged dict of cg_tolerance / cg_max_iters overrides.
    """
    relevant_keys = ("cg_tolerance", "cg_max_iters")
    merged: dict[str, Any] = {}
    for key in relevant_keys:
        value = getattr(generation_cfg, key, None)
        if value is not None:
            merged[key] = value
        if strategy and key in strategy.options:
            merged[key] = strategy.options[key]
    return merged


def _resolve_rhs_archive_glob(
    strategy: StrategySpec | None,
    context: DataGenerationContext,
    generation_cfg: GenerationConfig,
) -> str | None:
    """Resolve the RHS glob path to use for RHS archive samples.

    Tries in priority order: strategy rhs_glob, strategy rhs_path,
    generation-level rhs_archive_glob, context rhs_path.

    Args:
        strategy: Optional strategy spec with rhs_glob / rhs_path options.
        context: Generation context providing the fallback rhs_path.
        generation_cfg: Generation config providing the rhs_archive_glob fallback.

    Returns:
        Resolved glob string, or None if no source is available.
    """
    candidates: tuple[Any, ...] = (
        strategy.options.get("rhs_glob") if strategy else None,
        strategy.options.get("rhs_path") if strategy else None,
        generation_cfg.rhs_archive_glob,
        context.rhs_path,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _resolve_solution_archive_path(
    strategy: StrategySpec,
    context: DataGenerationContext,
) -> str | None:
    """Resolve the solutions glob/path for solution archive ingestion.

    Args:
        strategy: Strategy spec with solutions_glob / solutions_path options.
        context: Generation context providing the fallback solutions_path.

    Returns:
        Resolved solutions path string, or None if unavailable.
    """
    candidates: tuple[Any, ...] = (
        strategy.options.get("solutions_glob"),
        strategy.options.get("solutions_path"),
        context.solutions_path,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def compute_rhs_from_solution(
    *,
    matrix: np.ndarray,
    solutions_glob: str,
) -> np.ndarray:
    """Derive a fallback RHS vector by applying A @ x to a stored solution.

    Read-only: reads the lexicographically first solution matching the glob and
    returns the computed vector. Writing it is :func:`persist_derived_rhs`'s job.

    Used when ``provide_rhs=True`` and no explicit rhs_path is configured.

    Args:
        matrix: System matrix A used to compute A @ x.
        solutions_glob: Glob pattern pointing to solution files.

    Returns:
        The derived RHS vector.

    Raises:
        FileNotFoundError: If the solutions directory or files don't exist.
        ValueError: If no RHS could be loaded from the representative solution.
    """
    pattern = Path(solutions_glob)
    solution_dir = pattern.parent
    if not solution_dir.exists():
        raise FileNotFoundError(
            f"Solutions directory not found for provide_rhs automation: {solution_dir}"
        )

    candidates = sorted(solution_dir.glob(pattern.name))
    if not candidates:
        raise FileNotFoundError(
            "No solution files available to derive RHS for provide_rhs automation: "
            f"{solutions_glob}"
        )

    representative = candidates[0]
    samples = run_generation(
        "solution_archive",
        matrix,
        cfg={"solutions_glob": str(representative), "samples": 1},
        archive=None,
    )

    if samples.rhs is None or len(samples.rhs) == 0:
        raise ValueError(f"Failed to load solution from {representative}")

    return samples.rhs[0]


def persist_derived_rhs(rhs_vector: np.ndarray, *, dataset_dir: Path) -> Path:
    """Write a derived RHS vector into the dataset directory.

    Args:
        rhs_vector: The RHS vector to persist.
        dataset_dir: Directory where the derived RHS file is written; created
            along with any missing parents.

    Returns:
        Path to the written ``mother-rhs.txt`` file.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / "mother-rhs.txt"
    np.savetxt(output_path, rhs_vector, fmt="%.18e")
    return output_path


def _resolve_rhs_source(
    context: DataGenerationContext,
    solution_archive_options: Mapping[str, Any] | None,
    provide_rhs_option: Any,
    matrix: np.ndarray | None,
) -> str | None:
    """Determine the RHS path for synthetic generation, with provide_rhs overrides.

    Args:
        context: Generation context (rhs_path used as first priority).
        solution_archive_options: Options from the solution_archive strategy,
            required when provide_rhs=True.
        provide_rhs_option: Value of the provide_rhs config key — may be
            a bool, a path string, or None.
        matrix: System matrix, required when provide_rhs=True to compute A @ x.

    Returns:
        Resolved RHS path string, or None if no RHS source applies.

    Raises:
        FileNotFoundError: If a provide_rhs path doesn't exist.
        ValueError: If provide_rhs=True but required config is missing.
    """
    if context.rhs_path:
        return context.rhs_path

    if provide_rhs_option is None:
        return None

    if isinstance(provide_rhs_option, (str, Path)):
        path = Path(provide_rhs_option)
        if not path.exists():
            raise FileNotFoundError(f"provide_rhs path does not exist: {path}")
        return str(path)

    if isinstance(provide_rhs_option, bool) and provide_rhs_option:
        if matrix is None:
            raise ValueError(
                "provide_rhs=True is not supported with a matrix glob source; "
                "supply an explicit rhs_path instead."
            )
        if not solution_archive_options:
            raise ValueError(
                "provide_rhs=True requires solution archive options to resolve a solutions glob."
            )
        solutions_glob = solution_archive_options.get("solutions_glob")
        if not isinstance(solutions_glob, str) or not solutions_glob:
            raise ValueError(
                "provide_rhs=True requires 'solutions_glob' for the solution_archive strategy."
            )
        rhs_vector = compute_rhs_from_solution(matrix=matrix, solutions_glob=solutions_glob)
        return str(persist_derived_rhs(rhs_vector, dataset_dir=context.dataset_dir))

    return None
