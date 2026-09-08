"""Strategy execution dispatch for the generate workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from neuralls.composition.generation._archive_resolution import (
    _merge_rhs_archive_options,
    _resolve_rhs_archive_glob,
    _resolve_rhs_source,
    _resolve_solution_archive_path,
)
from neuralls.composition.generation._context_builder import DataGenerationContext
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.plan import GenerationPlan, StrategySpec
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec
from neuralls.shared.constants import DEFAULT_RANDOM_SEED


def _execute_solution_archive(
    context: DataGenerationContext,
    strategy: StrategySpec,
    generation_cfg: Any,
    *,
    force: bool = False,
) -> Path:
    """Execute a solution-archive-only dataset build.

    Args:
        context: Generation context with paths and options.
        strategy: StrategySpec for the solution_archive strategy.
        generation_cfg: GenerationConfig providing shuffle/seed defaults.

    Returns:
        Path to the generated dataset directory.

    Raises:
        ValueError: If no solutions glob can be resolved.
    """
    resolved_solutions_path = _resolve_solution_archive_path(strategy, context)
    if resolved_solutions_path is None:
        raise ValueError(
            "Solution archive strategy requires either "
            "a 'solutions_glob' field within [[generation.strategy]] name='solution_archive' or "
            "'source.solutions_path'."
        )

    shuffle_value = strategy.options.get("shuffle", generation_cfg.shuffle)
    seed_value = strategy.options.get("seed", generation_cfg.seed)

    dataset_path = build_dataset(
        context.source_spec(),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"solution_archive": strategy.samples},
                shuffle=bool(shuffle_value),
                seed=int(seed_value) if seed_value is not None else DEFAULT_RANDOM_SEED,
                strategy_overrides={
                    "solution_archive": {
                        "solutions_glob": resolved_solutions_path,
                        "shuffle": bool(shuffle_value),
                        "seed": seed_value,
                    }
                },
            ),
            replacement=context.replacement,
            normalize=context.normalize,
        ),
        str(context.dataset_dir),
        dataset_format=context.dataset_format,
        force=force,
    )
    return Path(dataset_path)


def _execute_synthetic_generation(
    context: DataGenerationContext,
    synthetic_strategies: Mapping[str, StrategySpec],
    rhs_archive_strategy: StrategySpec | None,
    solution_archive_strategy: StrategySpec | None,
    generation_cfg: Any,
    matrix: np.ndarray | None,
    *,
    force: bool = False,
) -> Path:
    """Execute a mixed synthetic + archive dataset build.

    Args:
        context: Generation context with paths and options.
        synthetic_strategies: Map of strategy name to StrategySpec for all
            non-archive strategies.
        rhs_archive_strategy: Optional RHS archive StrategySpec.
        solution_archive_strategy: Optional solution archive StrategySpec.
        generation_cfg: GenerationConfig providing global options.
        matrix: Optional loaded system matrix (required for provide_rhs derivation).

    Returns:
        Path to the generated dataset directory.

    Raises:
        ValueError: If a required RHS archive glob cannot be resolved.
    """
    seed_value = generation_cfg.seed
    if seed_value is None:
        seed_value = DEFAULT_RANDOM_SEED
    seed = int(seed_value)
    shuffle_value = generation_cfg.shuffle

    counts: dict[str, int] = {}
    strategy_overrides: dict[str, dict[str, Any]] = {}

    for name, spec in synthetic_strategies.items():
        counts[name] = spec.samples
        if spec.options:
            strategy_overrides[name] = dict(spec.options)

    if solution_archive_strategy is not None:
        counts["solution_archive"] = solution_archive_strategy.samples
        solutions_glob = _resolve_solution_archive_path(solution_archive_strategy, context)
        if solutions_glob:
            strategy_overrides["solution_archive"] = {
                "solutions_glob": solutions_glob,
                "shuffle": solution_archive_strategy.options.get("shuffle", generation_cfg.shuffle),
                "seed": solution_archive_strategy.options.get("seed", seed_value),
            }

    if rhs_archive_strategy is not None:
        counts["rhs_archive"] = rhs_archive_strategy.samples
        rhs_archive_glob = _resolve_rhs_archive_glob(rhs_archive_strategy, context, generation_cfg)
        if rhs_archive_glob is None:
            raise ValueError(
                "RHS archive strategy requires 'rhs_glob' within the strategy, "
                "'source.rhs_path', or 'generation.rhs_archive_glob'."
            )
        rhs_archive_opts = _merge_rhs_archive_options(rhs_archive_strategy, generation_cfg)
        strategy_overrides["rhs_archive"] = {"rhs_glob": rhs_archive_glob, **rhs_archive_opts}

    rhs_source_path = _resolve_rhs_source(
        context=context,
        solution_archive_options=solution_archive_strategy.options
        if solution_archive_strategy
        else None,
        provide_rhs_option=None,
        matrix=matrix,
    )

    dataset_path = build_dataset(
        context.source_spec(rhs_path=rhs_source_path),
        DatasetSpec(
            mixture=MixtureSpec(
                counts=counts,
                shuffle=bool(shuffle_value),
                seed=seed,
                strategy_overrides=strategy_overrides,
            ),
            replacement=context.replacement,
            normalize=context.normalize,
        ),
        str(context.dataset_dir),
        dataset_format=context.dataset_format,
        force=force,
    )
    return Path(dataset_path)


def _execute_rhs_archive_only(
    context: DataGenerationContext,
    strategy: StrategySpec,
    generation_cfg: Any,
    *,
    force: bool = False,
) -> Path:
    """Execute an RHS-archive-only dataset build.

    Args:
        context: Generation context with paths and options.
        strategy: StrategySpec for the rhs_archive strategy.
        generation_cfg: GenerationConfig providing global options.

    Returns:
        Path to the generated dataset directory.

    Raises:
        ValueError: If no RHS glob can be resolved.
    """
    rhs_glob = _resolve_rhs_archive_glob(strategy, context, generation_cfg)
    if rhs_glob is None:
        raise ValueError(
            "RHS archive strategy requires either 'rhs_glob' in the strategy or 'source.rhs_path'."
        )

    collection_kwargs = _merge_rhs_archive_options(strategy, generation_cfg)
    seed_value = generation_cfg.seed

    dataset_path = build_dataset(
        context.source_spec(),
        DatasetSpec(
            mixture=MixtureSpec(
                counts={"rhs_archive": strategy.samples},
                seed=int(seed_value) if seed_value is not None else DEFAULT_RANDOM_SEED,
                strategy_overrides={"rhs_archive": {"rhs_glob": rhs_glob, **collection_kwargs}},
            ),
            replacement=context.replacement,
            normalize=context.normalize,
        ),
        str(context.dataset_dir),
        dataset_format=context.dataset_format,
        force=force,
    )
    return Path(dataset_path)


def _execute_plan(
    context: DataGenerationContext,
    generation_cfg: Any,
    plan: GenerationPlan,
    matrix: np.ndarray | None,
    *,
    force: bool = False,
) -> Path:
    """Dispatch execution to the appropriate executor based on the generation plan.

    Args:
        context: Generation context with paths and options.
        generation_cfg: GenerationConfig providing global options.
        plan: Resolved generation plan (synthetic + archive strategies).
        matrix: Optional loaded system matrix.
        force: Regenerate even if a matching dataset already exists at
            `context.dataset_dir`.

    Returns:
        Path to the generated dataset directory.

    Raises:
        ValueError: If no strategies are configured.
    """
    rhs_archive_strategy = plan.rhs_archive
    solution_archive_strategy = plan.solution_archive
    synthetic_strategies = plan.synthetic

    has_solution_archive = solution_archive_strategy is not None
    has_rhs_archive = rhs_archive_strategy is not None
    has_synthetic_strategies = bool(synthetic_strategies)

    if solution_archive_strategy is not None and not (has_rhs_archive or has_synthetic_strategies):
        return _execute_solution_archive(
            context, solution_archive_strategy, generation_cfg, force=force
        )

    if has_synthetic_strategies or has_solution_archive:
        return _execute_synthetic_generation(
            context=context,
            synthetic_strategies=synthetic_strategies,
            rhs_archive_strategy=rhs_archive_strategy,
            solution_archive_strategy=solution_archive_strategy,
            generation_cfg=generation_cfg,
            matrix=matrix,
            force=force,
        )

    if rhs_archive_strategy is None:
        raise ValueError("No generation strategies configured")

    return _execute_rhs_archive_only(context, rhs_archive_strategy, generation_cfg, force=force)
