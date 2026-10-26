"""Scientific descriptions of generation strategies and of generated datasets.

``describe_strategy`` turns one validated strategy config into the facts that identify
its data (every algorithmic parameter, the effective seed as a reproducibility fact).
Facts that only say where files live or how rows are stored are deliberately absent.
``provenance.py`` assembles them for a whole dataset build. The names
come from the shared strategy-name table, so logs, manifests, and plots agree.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel

from neuralls.domain.generation.plan import canonicalize_strategy_kind
from neuralls.domain.generation.strategy_configs import (
    BaseEigenvectorConfig,
    BaseStrategyConfig,
    ConstantForwardConfig,
    ConstantInverseConfig,
    EigenvectorForwardConfig,
    EigenvectorInverseConfig,
    GaussianForwardConfig,
    GaussianInverseConfig,
    KrylovConfig,
    RandomNormalConfig,
    ResidualErrorConfig,
    RhsArchiveConfig,
    ScaledSolutionsConfig,
    SinusoidalFieldConfig,
    SmootherFilteredProbesConfig,
    SolutionArchiveConfig,
    SolveConfig,
    SparseRhsConfig,
    UniformForwardConfig,
    UniformInverseConfig,
    ValidatedArchiveConfig,
)
from neuralls.shared.scientific_description import (
    DAMPING,
    DecimalFact,
    FieldImportance,
    FieldKey,
    IntegerFact,
    ScientificDescription,
    ScientificFact,
    TextFact,
)
from neuralls.shared.scientific_description.catalog import (
    CONSTANT_VALUE,
    EIGENVALUE_SELECTION,
    EIGENVECTOR_COUNT,
    EIGENVECTOR_INCLUSION,
    FREQUENCY,
    ITERATION_COUNT,
    KRYLOV_DIMENSION,
    LOWER_BOUND,
    MEAN,
    PROBE_DISTRIBUTION,
    RHS_SCALE,
    SAMPLE_COUNT,
    SEED,
    SOLUTION_SCALE,
    SOLVER_ABSOLUTE_TOLERANCE,
    SOLVER_KIND,
    SOLVER_MAX_ITERATIONS,
    SOLVER_POSITIVE_DEFINITE,
    SOLVER_RELATIVE_TOLERANCE,
    SPARSE_SUPPORT,
    STANDARD_DEVIATION,
    STEPS,
    UPPER_BOUND,
    VERIFICATION_TOLERANCE,
    WINDOW_START,
    WINDOW_STRIDE,
    FactRole,
)
from neuralls.shared.scientific_description.presentation import (
    StrategyProvenance,
    strategy_base_name,
)
from neuralls.shared.types import GenerationStrategyKind

__all__ = [
    "FIELD_ROLES",
    "RESIDUAL_TRACE_STRATEGY_NAMES",
    "describe_generation_result",
    "describe_residual_trace",
    "describe_strategy",
]

_ESSENTIAL = FieldImportance.ESSENTIAL

RESIDUAL_TRACE_STRATEGY_NAMES = frozenset({"residuals", "gaussian_residuals"})
"""Strategy names ``runner.py`` must not pass to ``describe_generation_result`` — they
already log a richer description themselves from inside ``generate()``."""

type Facts = tuple[ScientificFact, ...]


_A = FactRole.ALGORITHMIC
_R = FactRole.REPRODUCIBILITY
_O = FactRole.OPERATIONAL
_P = FactRole.PROVENANCE
_BASE_ROLES = {"samples": _A, "seed": _R, "shuffle": _O}
_SOLVE = {"solve_config": _A}
_WINDOW = {"stop": _A, "start": _A, "step": _A}
_EIGEN = {"which": _A, "num_eigenvectors": _A, "include_eigenvectors": _A}

FIELD_ROLES: Mapping[type[BaseModel], Mapping[str, FactRole]] = MappingProxyType(
    {
        BaseStrategyConfig: _BASE_ROLES,
        BaseEigenvectorConfig: {**_BASE_ROLES, **_EIGEN},
        EigenvectorForwardConfig: {**_BASE_ROLES, **_EIGEN},
        EigenvectorInverseConfig: {**_BASE_ROLES, **_EIGEN, **_SOLVE},
        KrylovConfig: {**_BASE_ROLES, "krylov_iters": _A},
        SmootherFilteredProbesConfig: {
            **_BASE_ROLES,
            **_WINDOW,
            "omega": _A,
            "probe_distribution": _A,
        },
        RandomNormalConfig: {**_BASE_ROLES, "target_rhs_scale": _A},
        GaussianForwardConfig: {**_BASE_ROLES, "mu": _A, "sigma": _A},
        GaussianInverseConfig: {**_BASE_ROLES, "mu": _A, "sigma": _A, **_SOLVE},
        UniformForwardConfig: {**_BASE_ROLES, "a": _A, "b": _A},
        UniformInverseConfig: {**_BASE_ROLES, "a": _A, "b": _A, **_SOLVE},
        SinusoidalFieldConfig: {**_BASE_ROLES, "omega": _A, **_SOLVE},
        ConstantForwardConfig: {**_BASE_ROLES, "c": _A},
        ConstantInverseConfig: {**_BASE_ROLES, "c": _A, **_SOLVE},
        ResidualErrorConfig: {
            **_BASE_ROLES,
            **_WINDOW,
            "archive_rhs": _O,
            "archive_solutions": _O,
            "solutions_glob": _P,
            "file_indices": _O,
        },
        RhsArchiveConfig: {**_BASE_ROLES, "rhs_glob": _P, "file_indices": _O, **_SOLVE},
        SolutionArchiveConfig: {
            **_BASE_ROLES,
            "solutions_glob": _P,
            "file_indices": _O,
            "skip": _O,
        },
        ValidatedArchiveConfig: {
            **_BASE_ROLES,
            "solutions_glob": _P,
            "rhs_glob": _P,
            "file_indices": _O,
            "skip": _O,
            "verification_tolerance": _A,
            "fail_on_invalid": _O,
        },
        ScaledSolutionsConfig: {
            **_BASE_ROLES,
            "solutions_glob": _P,
            "file_indices": _O,
            "skip": _O,
            "scale": _A,
        },
        SparseRhsConfig: {**_BASE_ROLES, "indices": _P, "values": _P, **_SOLVE},
        SolveConfig: {
            "method": _A,
            "rtol": _A,
            "atol": _A,
            "max_iters": _A,
            "assume_pos_def": _A,
        },
    }
)
"""Every strategy-config field's role. A new field fails the exhaustiveness test until it is
classified, so it cannot silently drop out of a dataset's description. Archive paths and file
selection only say where rows come from, so they are provenance/operational, not facts;
a sparse RHS contributes its support size, while the index/value lists stay provenance."""


def _int(key: FieldKey, value: int) -> IntegerFact:
    return IntegerFact(key, value, _ESSENTIAL)


def _dec(key: FieldKey, value: float) -> DecimalFact:
    return DecimalFact(key, value, _ESSENTIAL)


def _on_off(key: FieldKey, value: bool) -> TextFact:
    return TextFact(key, "on" if value else "off", _ESSENTIAL)


def _solver_facts(solve: SolveConfig) -> Facts:
    if solve.method == "direct":
        return (TextFact(SOLVER_KIND, "direct", _ESSENTIAL),)
    return (
        TextFact(SOLVER_KIND, "cg", _ESSENTIAL),
        _dec(SOLVER_RELATIVE_TOLERANCE, solve.rtol),
        _dec(SOLVER_ABSOLUTE_TOLERANCE, solve.atol),
        _int(SOLVER_MAX_ITERATIONS, solve.max_iters),
        _on_off(SOLVER_POSITIVE_DEFINITE, solve.assume_pos_def),
    )


def _window_facts(
    cfg: ResidualErrorConfig | SmootherFilteredProbesConfig, stop_key: FieldKey
) -> Facts:
    return (
        _int(stop_key, cfg.stop),
        *(() if cfg.start is None else (_int(WINDOW_START, cfg.start),)),
        *(() if cfg.step == 1 else (_int(WINDOW_STRIDE, cfg.step),)),
    )


def _parameter_facts(cfg: BaseStrategyConfig) -> Facts:
    """Algorithmic facts of ``cfg`` in display order, excluding sample count and seed."""
    match cfg:
        case KrylovConfig():
            return (_int(KRYLOV_DIMENSION, cfg.krylov_iters),)
        case SmootherFilteredProbesConfig():
            return (
                *_window_facts(cfg, STEPS),
                TextFact(PROBE_DISTRIBUTION, cfg.probe_distribution, _ESSENTIAL),
                TextFact(DAMPING, "auto", _ESSENTIAL)
                if cfg.omega is None
                else _dec(DAMPING, cfg.omega),
            )
        case ResidualErrorConfig():
            return _window_facts(cfg, ITERATION_COUNT)
        case RandomNormalConfig():
            return (_dec(RHS_SCALE, cfg.target_rhs_scale),)
        case GaussianInverseConfig():
            return (
                _dec(MEAN, cfg.mu),
                _dec(STANDARD_DEVIATION, cfg.sigma),
                *_solver_facts(cfg.solve_config),
            )
        case GaussianForwardConfig():
            return (_dec(MEAN, cfg.mu), _dec(STANDARD_DEVIATION, cfg.sigma))
        case UniformInverseConfig():
            return (
                _dec(LOWER_BOUND, cfg.a),
                _dec(UPPER_BOUND, cfg.b),
                *_solver_facts(cfg.solve_config),
            )
        case UniformForwardConfig():
            return (_dec(LOWER_BOUND, cfg.a), _dec(UPPER_BOUND, cfg.b))
        case ConstantInverseConfig():
            return (_dec(CONSTANT_VALUE, cfg.c), *_solver_facts(cfg.solve_config))
        case ConstantForwardConfig():
            return (_dec(CONSTANT_VALUE, cfg.c),)
        case SinusoidalFieldConfig():
            return (_dec(FREQUENCY, cfg.omega), *_solver_facts(cfg.solve_config))
        case EigenvectorInverseConfig():
            return (*_eigenvector_facts(cfg), *_solver_facts(cfg.solve_config))
        case BaseEigenvectorConfig():
            return _eigenvector_facts(cfg)
        case RhsArchiveConfig():
            return _solver_facts(cfg.solve_config)
        case SparseRhsConfig():
            return (_int(SPARSE_SUPPORT, len(cfg.indices)), *_solver_facts(cfg.solve_config))
        case ScaledSolutionsConfig():
            return (_dec(SOLUTION_SCALE, cfg.scale),)
        case ValidatedArchiveConfig():
            return (_dec(VERIFICATION_TOLERANCE, cfg.verification_tolerance),)
        case _:
            return ()


def _eigenvector_facts(cfg: BaseEigenvectorConfig) -> Facts:
    return (
        TextFact(EIGENVALUE_SELECTION, cfg.which.value, _ESSENTIAL),
        _int(EIGENVECTOR_COUNT, cfg.num_eigenvectors),
        _on_off(EIGENVECTOR_INCLUSION, cfg.include_eigenvectors),
    )


def describe_strategy(
    kind: GenerationStrategyKind, cfg: BaseStrategyConfig, *, sample_count: int | None
) -> StrategyProvenance:
    """Describe one strategy: window/parameters, realized rows ``N`` when known, seed.

    ``sample_count`` is the number of rows the strategy produces, or ``None`` when it is
    open-ended (``samples=-1``) and not known per strategy.
    """
    facts = _parameter_facts(cfg)
    sample_facts: Facts = () if sample_count is None else (_int(SAMPLE_COUNT, sample_count),)
    # Trajectory strategies read "k=20, N=5000, ..." (the cap, then the count, then the rest);
    # every other strategy lists its parameters and then the count.
    leading = (
        1 if isinstance(cfg, ResidualErrorConfig | SmootherFilteredProbesConfig) else len(facts)
    )
    seed_facts: Facts = () if cfg.seed is None else (_int(SEED, cfg.seed),)
    return StrategyProvenance(
        kind=kind,
        facts=(*facts[:leading], *sample_facts, *facts[leading:], *seed_facts),
    )


def describe_residual_trace(
    strategy_name: str, *, iteration_count: int, sample_count: int
) -> ScientificDescription:
    """Realized description of one residual-trace strategy's generated rows: ``k`` and ``N``."""
    return ScientificDescription(
        base_name=strategy_base_name(canonicalize_strategy_kind(strategy_name)),
        facts=(_int(ITERATION_COUNT, iteration_count), _int(SAMPLE_COUNT, sample_count)),
    )


def describe_generation_result(strategy_name: str, *, sample_count: int) -> ScientificDescription:
    """Realized description of one strategy's generated rows for logging: name and ``N``.

    Raises:
        ValueError: If ``strategy_name`` is not a known generation strategy.
    """
    return ScientificDescription(
        base_name=strategy_base_name(canonicalize_strategy_kind(strategy_name)),
        facts=(_int(SAMPLE_COUNT, sample_count),),
    )
