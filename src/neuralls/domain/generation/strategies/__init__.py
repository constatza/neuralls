"""Data generation strategies."""

from . import (
    constant_strategies,
    eigenvector,
    gaussian_strategies,
    krylov,
    neutral_ones,
    random_normal,
    residuals,
    rhs_archive,
    scaled_solutions,
    search_directions,
    smoother_probes,
    solution_archive,
    sparse_rhs,
    uniform_strategies,
    validated_archive,
)

__all__ = [
    "constant_strategies",
    "eigenvector",
    "gaussian_strategies",
    "krylov",
    "neutral_ones",
    "random_normal",
    "residuals",
    "rhs_archive",
    "scaled_solutions",
    "search_directions",
    "smoother_probes",
    "solution_archive",
    "sparse_rhs",
    "uniform_strategies",
    "validated_archive",
]
