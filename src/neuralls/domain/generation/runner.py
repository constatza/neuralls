"""Registry and dispatcher for generation strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast, overload

import numpy as np
from loguru import logger

from neuralls.shared.scientific_description.policy import LOG_PROFILE, NotationPolicy
from neuralls.shared.scientific_description.render import render
from neuralls.shared.types import SystemMatrix

from .descriptions import RESIDUAL_TRACE_STRATEGY_NAMES, describe_generation_result
from .interfaces import (
    ArchiveData,
    GeneratedSamples,
    MatrixGenerationStrategy,
    SingleRhsGenerationStrategy,
    TracingSolverCallable,
)
from .strategy_configs import BaseStrategyConfig

RowsPerBaseSystem = Callable[[Mapping[str, Any]], int]
"""Maps a strategy's raw config to K_rows, the trajectory rows one base system yields."""


def _one_row_per_base_system(cfg: Mapping[str, Any]) -> int:
    """Strategies with no trajectory emit one row per base system."""
    del cfg
    return 1


@dataclass(frozen=True)
class MatrixStrategyRegistration:
    """Registered strategy entry for matrix-only generators."""

    strategy: MatrixGenerationStrategy
    supports_single_rhs: Literal[False] = False
    rows_per_base_system: RowsPerBaseSystem = _one_row_per_base_system


@dataclass(frozen=True)
class SingleRhsStrategyRegistration:
    """Registered strategy entry for single-RHS-capable generators."""

    strategy: SingleRhsGenerationStrategy
    supports_single_rhs: Literal[True] = True
    rows_per_base_system: RowsPerBaseSystem = _one_row_per_base_system


StrategyRegistration = MatrixStrategyRegistration | SingleRhsStrategyRegistration


class StrategyRegistry:
    """In-memory registry for generation strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyRegistration] = {}

    def register_matrix(
        self,
        strategy: MatrixGenerationStrategy,
        *,
        rows_per_base_system: RowsPerBaseSystem = _one_row_per_base_system,
    ) -> None:
        self._strategies[strategy.name] = MatrixStrategyRegistration(
            strategy,
            rows_per_base_system=rows_per_base_system,
        )

    def register_single_rhs(
        self,
        strategy: SingleRhsGenerationStrategy,
        *,
        rows_per_base_system: RowsPerBaseSystem = _one_row_per_base_system,
    ) -> None:
        self._strategies[strategy.name] = SingleRhsStrategyRegistration(
            strategy,
            rows_per_base_system=rows_per_base_system,
        )

    def get(self, name: str) -> StrategyRegistration:
        if name not in self._strategies:
            raise KeyError(f"Unknown generation strategy '{name}'")
        return self._strategies[name]


_registry = StrategyRegistry()


@overload
def register_strategy[StrategyClass](
    strategy_cls: type[StrategyClass],
    *,
    rows_per_base_system: RowsPerBaseSystem = ...,
) -> type[StrategyClass]: ...


@overload
def register_strategy[StrategyClass](
    strategy_cls: None = None,
    *,
    rows_per_base_system: RowsPerBaseSystem = ...,
) -> Callable[[type[StrategyClass]], type[StrategyClass]]: ...


def register_strategy[StrategyClass](
    strategy_cls: type[StrategyClass] | None = None,
    *,
    rows_per_base_system: RowsPerBaseSystem = _one_row_per_base_system,
) -> type[StrategyClass] | Callable[[type[StrategyClass]], type[StrategyClass]]:
    """Register a matrix-only generation strategy (see ``register_single_rhs_strategy``)."""

    def _decorate(cls: type[StrategyClass]) -> type[StrategyClass]:
        _registry.register_matrix(
            cast(MatrixGenerationStrategy, cls()),
            rows_per_base_system=rows_per_base_system,
        )
        return cls

    if strategy_cls is None:
        return _decorate
    return _decorate(strategy_cls)


def register_single_rhs_strategy[StrategyClass](
    strategy_cls: type[StrategyClass] | None = None,
    *,
    rows_per_base_system: RowsPerBaseSystem = _one_row_per_base_system,
) -> type[StrategyClass] | Any:
    """Register a generation strategy that supports shared RHS dispatch.

    ``rows_per_base_system`` gives K_rows for trajectory strategies so the orchestrator
    can convert a row budget into whole base systems. The default (one row per base
    system) suits strategies without a trajectory.
    """

    def _decorate(cls: type[StrategyClass]) -> type[StrategyClass]:
        _registry.register_single_rhs(
            cast(SingleRhsGenerationStrategy, cls()),
            rows_per_base_system=rows_per_base_system,
        )
        return cls

    if strategy_cls is None:
        return _decorate
    return _decorate(strategy_cls)


def rows_per_base_system(strategy_name: str, cfg: Mapping[str, Any]) -> int:
    """Return K_rows: trace rows one base system contributes to the strategy's output.

    Raises:
        KeyError: If the strategy name is unknown.
    """
    return _registry.get(strategy_name).rows_per_base_system(cfg)


def registered_config_type(strategy_name: str) -> type[BaseStrategyConfig]:
    """Return the validated config class a registered strategy parses its options with.

    Raises:
        KeyError: If the strategy name is unknown.
    """
    return cast(type[BaseStrategyConfig], _registry.get(strategy_name).strategy.ConfigType)


def run_generation(
    strategy_name: str,
    matrix: SystemMatrix,
    *,
    cfg: dict[str, Any],
    solver: TracingSolverCallable | None = None,
    archive: ArchiveData | None = None,
    single_rhs: np.ndarray | None = None,
) -> GeneratedSamples:
    """Execute a specific generation strategy.

    Dispatches to the appropriate strategy type, passing single_rhs only to strategies that support it.

    Args:
        strategy_name: Name of the strategy to run (must be registered)
        matrix: System matrix, shape (n, n)
        cfg: Strategy configuration dictionary (validated by strategy)
        solver: Tracing solver callable. Required for single-RHS (trace) strategies;
            neuralls does not construct a default solver of its own — inject one via
            ``neuralls.composition.generation.default_services.make_solver``.
        archive: Optional pre-loaded archive data to pass to the strategy
        single_rhs: Optional single RHS vector, shape (n,). If provided to single-RHS strategies
            (trace strategies), all samples will solve the same system A @ x = single_rhs

    Returns:
        GeneratedSamples containing matrix, rhs, solutions, and optional traces

    Raises:
        KeyError: If strategy name is unknown
        ValueError: If a single-RHS strategy is run without an explicit solver
    """
    registration = _registry.get(strategy_name)
    if isinstance(registration, SingleRhsStrategyRegistration):
        if solver is None:
            raise ValueError(
                f"Strategy '{strategy_name}' requires an explicit solver; neuralls does not "
                "construct a default one. Inject one via "
                "neuralls.composition.generation.default_services.make_solver()."
            )
        result = registration.strategy.generate(
            matrix,
            cfg=cfg,
            solver=solver,
            single_rhs=single_rhs,
            archive=archive,
        )
    else:
        result = registration.strategy.generate(matrix, cfg=cfg, archive=archive)
    _log_generation_result(strategy_name, result)
    return result


def _realized_row_count(result: GeneratedSamples) -> int:
    """The realized number of generated rows, from whichever field the strategy populated.

    ``smoother_filtered_probes`` populates only ``residual_traces`` (``rhs``
    and ``solutions`` are both ``None`` for it) — falling through to
    ``matrix.shape[0]`` there would log the system dimension instead of the
    realized sample count. ``error_traces`` is included for completeness
    even though its only producers (``residuals``/``gaussian_residuals``)
    never reach this function (see ``RESIDUAL_TRACE_STRATEGY_NAMES``).
    """
    if result.rhs is not None:
        return result.rhs.shape[0]
    if result.solutions is not None:
        return result.solutions.shape[0]
    if result.residual_traces is not None:
        return result.residual_traces.residuals.shape[0]
    if result.error_traces is not None:
        return result.error_traces.errors.shape[0]
    return result.matrix.shape[0]


def _log_generation_result(strategy_name: str, result: GeneratedSamples) -> None:
    """Log the realized scientific description for every strategy except the residual-trace pair.

    ``residuals``/``gaussian_residuals`` already log their own richer
    (``k``+``N``) description from inside ``generate()`` before this ever
    runs — see ``descriptions.py``'s module docstring.
    """
    if strategy_name in RESIDUAL_TRACE_STRATEGY_NAMES:
        return
    description = describe_generation_result(
        strategy_name, sample_count=_realized_row_count(result)
    )
    logger.info(render(description, LOG_PROFILE, NotationPolicy.default()).text)
