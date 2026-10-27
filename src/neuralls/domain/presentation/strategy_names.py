"""Controlled display names of generation strategies."""

from __future__ import annotations

from typing import assert_never

from neuralls.shared.types import GenerationStrategyKind


def strategy_base_name(kind: GenerationStrategyKind) -> str:
    """Return the controlled display name of a generation strategy."""
    match kind:
        case GenerationStrategyKind.RANDOM | GenerationStrategyKind.NORMAL:
            return "Random Normal Solutions"
        case GenerationStrategyKind.KRYLOV:
            return "Krylov Subspace Solutions"
        case GenerationStrategyKind.RHS_ARCHIVE:
            return "Archived RHS Solves"
        case GenerationStrategyKind.SOLUTION_ARCHIVE:
            return "Archived Solutions"
        case GenerationStrategyKind.VALIDATED_ARCHIVE:
            return "Validated Archive Pairs"
        case GenerationStrategyKind.SCALED_SOLUTIONS:
            return "Scaled Archived Solutions"
        case GenerationStrategyKind.SPARSE_RHS:
            return "Sparse RHS Solves"
        case GenerationStrategyKind.RESIDUALS:
            return "CG Residuals"
        case GenerationStrategyKind.GAUSSIAN_RESIDUALS:
            return "Gaussian CG Residuals"
        case GenerationStrategyKind.EIGENVECTOR_FORWARD:
            return "Eigenvector Solutions"
        case GenerationStrategyKind.EIGENVECTOR_INVERSE:
            return "Eigenvector RHS Solves"
        case GenerationStrategyKind.GAUSSIAN_FORWARD:
            return "Gaussian Solutions"
        case GenerationStrategyKind.GAUSSIAN_INVERSE:
            return "Gaussian RHS Solves"
        case GenerationStrategyKind.UNIFORM_FORWARD:
            return "Uniform Solutions"
        case GenerationStrategyKind.UNIFORM_INVERSE:
            return "Uniform RHS Solves"
        case GenerationStrategyKind.CONSTANT_FORWARD:
            return "Constant Solutions"
        case GenerationStrategyKind.CONSTANT_INVERSE:
            return "Constant RHS Solves"
        case GenerationStrategyKind.NEUTRAL_ONES:
            return "Ones Solutions"
        case GenerationStrategyKind.SMOOTHER_FILTERED_PROBES:
            return "Jacobi-Filtered Probes"
        case GenerationStrategyKind.SINUSOIDAL_FIELD:
            return "Sinusoidal RHS Solves"
        case _:
            assert_never(kind)
