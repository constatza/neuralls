"""Scientific descriptions for generation strategies (``docs/plan.md`` Phase 5).

Every registered strategy name (see ``domain/generation/strategies/*.py``'s
``name = "..."`` class attributes) has a describer here:
``describe_residual_trace`` for the residual-trace family (``k``+``N``, the
plan's own worked example), ``describe_generation_result`` as the single
dispatch point for every other strategy (``N`` only — none of the other
strategies have a second structural parameter as central as CG's iteration
cap). ``runner.py::run_generation`` calls ``describe_generation_result`` for
every strategy except the residual-trace pair, which already log their own
richer description from inside ``generate()`` before this dispatcher would
ever see them.
"""

from __future__ import annotations

from neuralls.shared.scientific_description import (
    FieldImportance,
    FieldKey,
    IntegerFact,
    ScientificDescription,
)

ITERATION_COUNT = FieldKey("iteration_count")
SAMPLE_COUNT = FieldKey("sample_count")

_RESIDUAL_TRACE_BASE_NAMES: dict[str, str] = {
    "residuals": "CG Residuals",
    "gaussian_residuals": "Gaussian CG Residuals",
}

RESIDUAL_TRACE_STRATEGY_NAMES = frozenset(_RESIDUAL_TRACE_BASE_NAMES)
"""Strategy names ``runner.py`` must not pass to ``describe_generation_result``
— they already log a richer description themselves (see module docstring)."""

_GENERATION_RESULT_BASE_NAMES: dict[str, str] = {
    "random": "Gaussian LHS",
    "normal": "Gaussian LHS",
    "krylov": "Krylov Combinations",
    "rhs_archive": "RHS Archive",
    "solution_archive": "Solution Archive",
    "validated_archive": "Validated Archive",
    "scaled_solutions": "Scaled Solutions",
    "sparse_rhs": "Sparse RHS",
    "eigenvector_forward": "Eigenvector Forward",
    "eigenvector_inverse": "Eigenvector Inverse",
    "gaussian_forward": "Gaussian Forward",
    "gaussian_inverse": "Gaussian Inverse",
    "uniform_forward": "Uniform Forward",
    "uniform_inverse": "Uniform Inverse",
    "constant_forward": "Constant Forward",
    "constant_inverse": "Constant Inverse",
    "neutral_ones": "Neutral Ones",
    "smoother_filtered_probes": "Smoother-Filtered Probes",
    "sinusoidal_field": "Sinusoidal Field",
}


def describe_residual_trace(
    strategy_name: str, *, iteration_count: int, sample_count: int
) -> ScientificDescription:
    """Realized description of one residual-trace strategy's generated rows.

    Args:
        strategy_name: The registered strategy name (``"residuals"`` or
            ``"gaussian_residuals"``); falls back to the raw name as the
            base name for any other caller.
        iteration_count: The CG iteration cap (``window.stop``) — fixed by
            configuration, not realized data.
        sample_count: The realized number of generated trajectory rows
            actually returned (``referenced_samples``), never the
            configured base-system count — the two differ whenever a
            strategy's window keeps more than one step per base system
            (see ``docs/plan.md``: "Trace counts use realized generated
            rows as N, not base-system counts").

    Returns:
        ScientificDescription: ``"{base_name} (k={iteration_count},
            N={sample_count})"`` once rendered.
    """
    base_name = _RESIDUAL_TRACE_BASE_NAMES.get(strategy_name, strategy_name)
    return ScientificDescription(
        base_name=base_name,
        facts=(
            IntegerFact(
                key=ITERATION_COUNT, value=iteration_count, importance=FieldImportance.ESSENTIAL
            ),
            IntegerFact(key=SAMPLE_COUNT, value=sample_count, importance=FieldImportance.ESSENTIAL),
        ),
    )


def describe_generation_result(strategy_name: str, *, sample_count: int) -> ScientificDescription:
    """Realized description of one strategy's generated rows: just ``N``.

    The single dispatch point for every registered strategy except
    ``residuals``/``gaussian_residuals`` (see ``RESIDUAL_TRACE_STRATEGY_NAMES``
    and the module docstring) — none of the other strategies have a second
    structural parameter as central to their identity as CG's iteration cap,
    so ``N`` (the realized row count) is the only fact rendered here.

    Args:
        strategy_name: The registered strategy name; falls back to the raw
            name as the base name for any strategy not in the known table
            (e.g. a new strategy added without updating
            ``_GENERATION_RESULT_BASE_NAMES`` yet).
        sample_count: The realized number of generated rows actually
            returned, never a configured ``-1``/"all" placeholder.

    Returns:
        ScientificDescription: ``"{base_name} (N={sample_count})"`` once
            rendered.
    """
    base_name = _GENERATION_RESULT_BASE_NAMES.get(strategy_name, strategy_name)
    return ScientificDescription(
        base_name=base_name,
        facts=(
            IntegerFact(key=SAMPLE_COUNT, value=sample_count, importance=FieldImportance.ESSENTIAL),
        ),
    )
