"""Scientific descriptions of generation strategies and of generated datasets.

``describe_strategy`` turns one validated strategy config into the facts that identify
its data (every algorithmic parameter, the realized row count and the seed).
Facts that only say where files live or how rows are stored are deliberately absent.
``provenance.py`` assembles them for a whole dataset build. The names
come from the shared strategy-name table, so logs, manifests, and plots agree.
"""

from __future__ import annotations

from neuralls.domain.generation.plan import canonicalize_strategy_kind
from neuralls.domain.generation.strategy_configs import BaseStrategyConfig
from neuralls.domain.presentation.provenance import StrategyProvenance, StrategySeed
from neuralls.domain.presentation.strategy_names import strategy_base_name
from neuralls.shared.fact_binding import describe_model
from neuralls.shared.scientific_description.catalog import ITERATION_COUNT, SAMPLE_COUNT, make_fact
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.types import GenerationStrategyKind

__all__ = [
    "RESIDUAL_TRACE_STRATEGY_NAMES",
    "describe_generation_result",
    "describe_residual_trace",
    "describe_strategy",
]


RESIDUAL_TRACE_STRATEGY_NAMES = frozenset({"residuals", "gaussian_residuals"})
"""Strategy names ``runner.py`` must not pass to ``describe_generation_result`` — they
already log a richer description themselves from inside ``generate()``."""


def describe_strategy(
    kind: GenerationStrategyKind, cfg: BaseStrategyConfig, *, rows: int, seed: StrategySeed
) -> StrategyProvenance:
    """Describe one strategy: its algorithmic parameters, the rows it produced, its seed.

    ``rows`` is the strategy's realized row count and ``seed`` says whether the
    configuration fixed the strategy's seed or each binding derived it; neither comes
    from ``cfg``, whose ``samples`` is a budget and whose ``seed`` has a default.
    """
    return StrategyProvenance(kind=kind, rows=rows, parameters=describe_model(cfg), seed=seed)


def describe_residual_trace(
    strategy_name: str, *, iteration_count: int, sample_count: int
) -> ScientificDescription:
    """Realized description of one residual-trace strategy's generated rows: ``k`` and ``N``."""
    return ScientificDescription(
        base_name=strategy_base_name(canonicalize_strategy_kind(strategy_name)),
        facts=(make_fact(ITERATION_COUNT, iteration_count), make_fact(SAMPLE_COUNT, sample_count)),
    )


def describe_generation_result(strategy_name: str, *, sample_count: int) -> ScientificDescription:
    """Realized description of one strategy's generated rows for logging: name and ``N``.

    Raises:
        ValueError: If ``strategy_name`` is not a known generation strategy.
    """
    return ScientificDescription(
        base_name=strategy_base_name(canonicalize_strategy_kind(strategy_name)),
        facts=(make_fact(SAMPLE_COUNT, sample_count),),
    )
