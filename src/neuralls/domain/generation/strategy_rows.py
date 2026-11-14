"""One strategy's generated training rows for one system, with row-kind classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import ValidationError

from neuralls.shared.enum_codecs import encode_row_kind_array
from neuralls.shared.types import GenerationStrategyKind, RowKind, SystemMatrix

from .interfaces import (
    ArchiveData,
    ErrorTraceRows,
    GeneratedSamples,
    ResidualTraceRows,
    SolvedRows,
)
from .notices import SolveNotice
from .ports import ArchiveSource
from .runner import run_generation
from .seeds import derive_strategy_seed
from .semantics import classify_strategy_row_kind
from .specs import MixtureSpec


@dataclass(frozen=True, slots=True)
class _StrategyRows:
    """One strategy's training pairs on one system, in generation order.

    Attributes:
        rhs: Feature rows (the RHS, or the trace residuals), shape (rows, n).
        solutions: Target rows (the solutions, or the trace errors), shape (rows, n).
        row_kind_codes: Per-row ``RowKind`` codes, shape (rows,).
        notices: Outcomes the strategy reported while producing the rows.
    """

    rhs: np.ndarray
    solutions: np.ndarray
    row_kind_codes: np.ndarray
    notices: tuple[SolveNotice, ...] = ()


def _row_kind_codes(
    strategy_name: str, row_count: int, iteration_indices: np.ndarray | None
) -> np.ndarray:
    """Classify each emitted row, using the trace iteration indices of a trace strategy."""
    if row_count == 0:
        return encode_row_kind_array([])
    base_kind = classify_strategy_row_kind(GenerationStrategyKind(strategy_name))
    # iter 0 is STANDARD only because CG initialises at x_0 = 0:
    # r_0 = b - A@0 = b  and  e_0 = x_true - 0 = x_true → (r_0, e_0) = (b, x_true).
    # WARNING: if the solver ever uses a non-zero initial guess this breaks silently.
    if iteration_indices is not None:
        return encode_row_kind_array(
            [RowKind.STANDARD if i == 0 else base_kind for i in iteration_indices]
        )
    return encode_row_kind_array([base_kind] * row_count)


def _rows_of(strategy_name: str, generated: GeneratedSamples) -> _StrategyRows:
    """The training pairs of one strategy result, whatever its row shape."""
    match generated:
        case ErrorTraceRows(error_traces=traces):
            return _StrategyRows(
                rhs=traces.residuals,  # r_k
                solutions=traces.errors,  # e_k = x_true - x_k
                row_kind_codes=_row_kind_codes(
                    strategy_name, generated.row_count, traces.iteration_indices
                ),
            )
        case ResidualTraceRows(residual_traces=traces):
            return _StrategyRows(
                rhs=traces.residuals,  # A @ p_k
                solutions=traces.solutions,  # p_k
                row_kind_codes=_row_kind_codes(
                    strategy_name, generated.row_count, traces.iteration_indices
                ),
            )
        case SolvedRows(rhs=rhs, solutions=solutions, notices=notices):
            return _StrategyRows(
                rhs=rhs,
                solutions=solutions,
                row_kind_codes=_row_kind_codes(strategy_name, generated.row_count, None),
                notices=notices,
            )


def _generate_strategy_rows(
    A: SystemMatrix,
    mixture: MixtureSpec,
    strategy_name: str,
    count: int,
    *,
    archive_solutions: np.ndarray | None = None,
    archive_rhs: np.ndarray | None = None,
    single_rhs: np.ndarray | None = None,
    single_solution: np.ndarray | None = None,
    archive_source: ArchiveSource | None = None,
) -> _StrategyRows:
    """Run one strategy for ``count`` samples and return its pairs and row kinds.

    Args:
        A: System matrix the strategy solves against, shape (n, n).
        mixture: Strategy overrides, solver overrides and the seed the strategy seed derives from.
        strategy_name: Registered strategy to run.
        count: Sample budget passed to the strategy as its ``samples`` option.
        archive_solutions: Pre-computed solutions for archive-based generation.
        archive_rhs: Pre-computed RHS vectors for archive-based generation.
        single_rhs: Single RHS vector for single-RHS (trace) strategies.
        single_solution: Single pre-computed solution used as a one-row archive.
        archive_source: Lists and loads the archive files of archive-backed strategies.

    Returns:
        The strategy's rows and the notices it reported.

    Raises:
        ValueError: If the strategy configuration is invalid.
    """
    cfg = dict((mixture.strategy_overrides or {}).get(strategy_name, {}))
    cfg.setdefault("seed", derive_strategy_seed(mixture.seed, strategy_name))
    cfg["samples"] = count

    effective_archive_solutions = archive_solutions
    if single_solution is not None and archive_solutions is None:
        effective_archive_solutions = single_solution.reshape(1, -1)

    archive_data: ArchiveData | None = None
    if effective_archive_solutions is not None:
        archive_data = ArchiveData(lhs=effective_archive_solutions, rhs=archive_rhs)

    # Pydantic (extra="forbid") will raise ValidationError on unknown keys — fail fast.
    try:
        generated = run_generation(
            strategy_name,
            A,
            cfg=cfg,
            solver=(
                mixture.solver_overrides.get(strategy_name) if mixture.solver_overrides else None
            ),
            archive_source=archive_source,
            archive=archive_data,
            single_rhs=single_rhs,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid configuration for strategy '{strategy_name}': {e}") from e

    return _rows_of(strategy_name, generated)
