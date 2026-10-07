"""Re-export surface for the generation package's split-out pure helper modules.

Kept so existing imports of ``neuralls.domain.generation.helpers`` keep working; new
code should import directly from the focused module (``seeds``, ``counts``, ``scaling``,
``linear_solve``, ``eigen_strategies``, ``archive_files``, ``krylov``).
"""

from __future__ import annotations

from .archive_files import select_archive_files, solution_row_count
from .counts import (
    _build_trace_indices,
    required_trace_systems,
    resolve_strategy_counts,
    resolve_trace_generation_counts,
    rounded_counts,
    trace_rows_per_base_system,
    trace_rows_per_system,
)
from .eigen_strategies import (
    _compute_eigendecomposition,
    _generate_eigenvector_combinations,
    _select_eigenvectors,
)
from .krylov import _generate_krylov_combinations, _lanczos_iteration
from .linear_solve import _solve_linear_systems, _verify_solution_accuracy
from .scaling import (
    _calculate_normalization_scale,
    normalize_matrix_for_generation,
    serialize_scale_metadata,
)
from .seeds import derive_seed, derive_strategy_seed, rng_from_seed

__all__ = [
    "_build_trace_indices",
    "_calculate_normalization_scale",
    "_compute_eigendecomposition",
    "_generate_eigenvector_combinations",
    "_generate_krylov_combinations",
    "_lanczos_iteration",
    "_select_eigenvectors",
    "_solve_linear_systems",
    "_verify_solution_accuracy",
    "derive_seed",
    "derive_strategy_seed",
    "normalize_matrix_for_generation",
    "required_trace_systems",
    "resolve_strategy_counts",
    "resolve_trace_generation_counts",
    "rng_from_seed",
    "rounded_counts",
    "select_archive_files",
    "serialize_scale_metadata",
    "solution_row_count",
    "trace_rows_per_base_system",
    "trace_rows_per_system",
]
