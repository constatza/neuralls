"""Operator log of a generation run, from the values the domain returns.

The domain reports outcomes as data (``GenerationNotice``, ``ScaleSummary.disagreements``,
``StrategyRowCount``); this module is the one place that turns them into log lines.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

from loguru import logger

from neuralls.domain.generation.batch import SampleBatch
from neuralls.domain.generation.descriptions import describe_generation_result
from neuralls.domain.generation.notices import (
    GenerationNotice,
    InvalidPairs,
    RowsCapped,
    SolveNotConverged,
)
from neuralls.domain.generation.orchestration import BatchStream
from neuralls.domain.generation.sample_writer import StrategyRowCount
from neuralls.domain.generation.scalar_aggregate import ScaleField, ScaleSummary
from neuralls.shared.scientific_description.policy import LOG_PROFILE, NotationPolicy
from neuralls.shared.scientific_description.render import render

_SCALE_DISAGREEMENT_TEXT: Final[dict[ScaleField, str]] = {
    ScaleField.MATRIX_NORM: (
        "Bindings produced different normalized matrix norms; "
        "the dataset manifest keeps a representative matrix_norm only."
    ),
    ScaleField.MATRIX_VALUE_SCALE: (
        "Bindings produced different matrix value scales; "
        "the dataset manifest omits a shared reversible matrix scale."
    ),
    ScaleField.SCALE_METADATA: (
        "Bindings produced different scale metadata payloads; "
        "the dataset manifest intentionally stores no shared scale metadata."
    ),
}


def render_notice(strategy: str, binding_index: int | None, notice: GenerationNotice) -> str:
    """The log line of one notice, naming the strategy and, when known, its binding."""
    where = f"Strategy '{strategy}'" + (
        f" (binding {binding_index})" if binding_index is not None else ""
    )
    match notice:
        case RowsCapped(requested_rows=requested, emitted_rows=emitted, source=source):
            return f"{where} requested {requested} rows but its {source} hold {emitted}: emitting {emitted} rows."
        case SolveNotConverged(row=row, exit_code=exit_code, residual=residual):
            return (
                f"{where}: CG did not converge on row {row} "
                f"(exit_code={exit_code}, residual: {residual:.2e})."
            )
        case InvalidPairs(count=count, total=total, max_error=max_error, tolerance=tolerance):
            return (
                f"{where}: found {count}/{total} invalid pairs "
                f"(max error: {max_error:.2e} > tolerance: {tolerance:.2e}); kept."
            )


def log_planned_notices(stream: BatchStream) -> None:
    """Log the notices known when the run was planned (rows capped by their source)."""
    for notice in stream.notices:
        match notice:
            case RowsCapped(strategy=strategy):
                logger.warning(render_notice(strategy, None, notice))
            case SolveNotConverged() | InvalidPairs():
                raise TypeError(f"{type(notice).__name__} cannot be known before generation")


def logged_batches(stream: BatchStream) -> Iterator[SampleBatch]:
    """Yield the stream's batches, logging each one and the notices it carries."""
    for batch in stream.batches:
        logger.info(
            f"Strategy '{batch.strategy_name}' (binding {batch.binding_index}): "
            f"{len(batch)} rows generated."
        )
        for notice in batch.notices:
            logger.warning(render_notice(batch.strategy_name, batch.binding_index, notice))
        yield batch


def logged_scale_summary(stream: BatchStream) -> ScaleSummary:
    """Resolve the run's scale summary and log each field the bindings disagreed on."""
    summary = stream.scale.result()
    for field in ScaleField:
        if field in summary.disagreements:
            logger.warning(_SCALE_DISAGREEMENT_TEXT[field])
    return summary


def log_strategy_rows(rows: tuple[StrategyRowCount, ...]) -> None:
    """Log the realized description of every strategy, once, after the run."""
    for row in rows:
        description = describe_generation_result(row.strategy_name, sample_count=row.rows)
        logger.info(render(description, LOG_PROFILE, NotationPolicy.default()).text)


__all__ = [
    "log_planned_notices",
    "log_strategy_rows",
    "logged_batches",
    "logged_scale_summary",
    "render_notice",
]
