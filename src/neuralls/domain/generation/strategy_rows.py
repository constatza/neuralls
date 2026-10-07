"""One strategy's generated training rows for one system, with row-kind classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import ValidationError

from neuralls.domain.normalization import ErrorTraceSamples, ResidualTraceSamples
from neuralls.shared.enum_codecs import encode_row_kind_array
from neuralls.shared.types import GenerationStrategyKind, RowKind, SystemMatrix

from .interfaces import ArchiveData
from .runner import GeneratedSamples, run_generation
from .seeds import derive_strategy_seed
from .semantics import classify_strategy_row_kind
from .specs import MixtureSpec


@dataclass(frozen=True)
class _StrategyRows:
    """One strategy's training pairs on one system, in generation order.

    Attributes:
        rhs: Feature rows (the RHS, or the trace residuals), shape (rows, n).
        solutions: Target rows (the solutions, or the trace errors), shape (rows, n).
        row_kind_codes: Per-row ``RowKind`` codes, shape (rows,).
    """

    rhs: np.ndarray
    solutions: np.ndarray
    row_kind_codes: np.ndarray


def _row_kind_codes_for(strategy_name: str, generated: GeneratedSamples) -> np.ndarray:
    """Classify each row a strategy emitted, using its trace iteration indices when present."""
    match generated:
        case GeneratedSamples(
            error_traces=ErrorTraceSamples(errors=errors, iteration_indices=iter_indices)
        ):
            semantic_size = int(errors.shape[0])
        case GeneratedSamples(
            residual_traces=ResidualTraceSamples(
                residuals=residuals, iteration_indices=iter_indices
            )
        ):
            semantic_size = int(residuals.shape[0])
        case GeneratedSamples(rhs=np.ndarray() as rhs):
            semantic_size = int(rhs.shape[0])
            iter_indices = None
        case _:
            semantic_size = 0
            iter_indices = None

    if semantic_size == 0:
        return encode_row_kind_array([])

    base_kind = classify_strategy_row_kind(GenerationStrategyKind(strategy_name))
    # iter 0 is STANDARD only because CG initialises at x_0 = 0:
    # r_0 = b - A@0 = b  and  e_0 = x_true - 0 = x_true → (r_0, e_0) = (b, x_true).
    # WARNING: if the solver ever uses a non-zero initial guess this breaks silently.
    if iter_indices is not None:
        row_kinds = [RowKind.STANDARD if i == 0 else base_kind for i in iter_indices]
    else:
        row_kinds = [base_kind] * semantic_size
    return encode_row_kind_array(row_kinds)


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
) -> _StrategyRows | None:
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

    Returns:
        The strategy's rows, or ``None`` when it produced neither pairs nor traces.

    Raises:
        ValueError: If the strategy configuration is invalid or its rhs/solutions disagree.
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
            archive=archive_data,
            single_rhs=single_rhs,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid configuration for strategy '{strategy_name}': {e}") from e

    row_kind_codes = _row_kind_codes_for(strategy_name, generated)
    # Trace strategies expose their training pairs via trace structs, not rhs/solutions.
    if generated.error_traces is not None:
        return _StrategyRows(
            rhs=generated.error_traces.residuals,  # r_k
            solutions=generated.error_traces.errors,  # e_k = x_true - x_k
            row_kind_codes=row_kind_codes,
        )
    if generated.residual_traces is not None:
        return _StrategyRows(
            rhs=generated.residual_traces.residuals,  # A @ p_k
            solutions=generated.residual_traces.solutions,  # p_k
            row_kind_codes=row_kind_codes,
        )
    if (generated.rhs is None) != (generated.solutions is None):
        raise ValueError(
            f"Strategy '{strategy_name}' returned rhs and solutions with inconsistent "
            f"None-ness: rhs={'None' if generated.rhs is None else 'array'}, "
            f"solutions={'None' if generated.solutions is None else 'array'}."
        )
    if generated.rhs is None or generated.solutions is None:
        return None
    return _StrategyRows(
        rhs=generated.rhs,
        solutions=generated.solutions,
        row_kind_codes=row_kind_codes,
    )
