"""Pure utility functions for data generation strategies."""

from __future__ import annotations

import math
import warnings
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
from loguru import logger
from scipy.linalg import norm

from neuralls.domain.normalization import IScale
from neuralls.shared.types import EigenvectorSelection, MatrixFormat, ScaleMetadata, SystemMatrix

from .matrix_operator import MatrixOperator, require_eigen_count
from .step_window import StepWindow


def rng_from_seed(seed: int | None) -> np.random.Generator:
    """Create random number generator from seed.

    Args:
        seed: Random seed (None for random)

    Returns:
        Random number generator
    """
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def derive_seed(mixture_seed: int, *labels: str | int) -> int:
    """Derive an independent, deterministic seed for one labelled random stream.

    Every stream that must not repeat another one (a strategy, a binding, a
    split level) takes its own label path under the mixture seed. String labels
    are hashed with CRC32, which is stable across processes unlike ``hash()``, so
    the same label path always maps to the same stream.

    Args:
        mixture_seed: Seed of the whole mixture.
        *labels: Stream discriminators, strings or non-negative integers.

    Returns:
        Non-negative integer seed for the labelled stream.
    """
    entropy = [mixture_seed, *(_label_entropy(label) for label in labels)]
    state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _label_entropy(label: str | int) -> int:
    """Map one stream label to a SeedSequence entropy word."""
    return zlib.crc32(label.encode("utf-8")) if isinstance(label, str) else label


def derive_strategy_seed(mixture_seed: int | None, strategy_name: str) -> int | None:
    """Derive an independent, deterministic seed for one strategy in a mixture.

    Strategies in one mixture must not share a random stream: with a shared
    seed, two strategies on the same matrix emit identical RHS vectors.

    Args:
        mixture_seed: Seed of the whole mixture (None keeps non-deterministic draws).
        strategy_name: Registered strategy name used as the stream discriminator.

    Returns:
        Non-negative integer seed for the strategy, or None when mixture_seed is None.
    """
    if mixture_seed is None:
        return None
    return derive_seed(mixture_seed, strategy_name)


def rounded_counts(total: int, proportions: Mapping[str, float]) -> dict[str, int]:
    """Convert strategy proportions to integer counts that sum exactly to total.

    The proportions do not need to sum to 1.0; they are normalized internally.

    Args:
        total: Total number of samples to generate
        proportions: Mapping of strategy names to proportions (non-negative)

    Returns:
        Dictionary mapping strategy names to integer counts

    Raises:
        ValueError: If total <= 0, proportions empty, or weights negative/zero sum
    """
    if total <= 0:
        raise ValueError(f"Total samples must be positive, got {total}")
    if not proportions:
        raise ValueError("At least one strategy proportion must be provided")

    weights: dict[str, float] = {}
    for key, value in proportions.items():
        weight = float(value)
        if weight < 0:
            raise ValueError(f"Mix weight for '{key}' must be non-negative, got {value}")
        weights[key] = weight

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Sum of mix weights must be positive")

    scaled = {key: (weight / total_weight) * total for key, weight in weights.items()}
    counts = {key: math.floor(amount) for key, amount in scaled.items()}
    remainders = {key: scaled[key] - counts[key] for key in scaled}

    remaining = total - sum(counts.values())
    if remaining > 0:
        sorted_keys = sorted(
            remainders.keys(),
            key=lambda key: (-remainders[key], -weights[key], key),
        )
        idx = 0
        while remaining > 0 and sorted_keys:
            key = sorted_keys[idx % len(sorted_keys)]
            counts[key] += 1
            remaining -= 1
            idx += 1

    return counts


def _calculate_normalization_scale(
    A: np.ndarray,
    b: np.ndarray,
    normalize: str,
) -> float:
    """Calculate target RHS scale based on normalization strategy.

    Pure function.

    Args:
        A: System matrix
        b: Mother RHS vector
        normalize: Normalization strategy

    Returns:
        Target RHS scale for generation
    """
    if normalize == "rhs":
        n = A.shape[0]
        target_norm = float(norm(b))
        return target_norm / np.sqrt(n)
    return 1.0


def normalize_matrix_for_generation(
    matrix: SystemMatrix,
    normalize_type: Literal["none", "matrix", "rhs"],
    spectral_radius_bound: float | None,
) -> tuple[SystemMatrix, IScale | None, float]:
    """Normalize matrix for synthetic generation (pure function).

    CONTRACT:
        - Input: Raw matrix A
        - Output: Normalized matrix A_norm and optional scale metadata
        - Strategies receive normalized matrix and compute b_norm = A_norm @ x

    Args:
        matrix: Raw system matrix A, dense or CSR. CSR input stays CSR; it is
            never densified.
        normalize_type: Normalization strategy
            - "none": No normalization (identity)
            - "matrix": Scale by spectral_radius_bound * sqrt(d)
            - "rhs": Legacy, treated as "none" (RHS matching handled by caller)
        spectral_radius_bound: For matrix normalization (computed if None)

    Returns:
        Tuple of (normalized_matrix, scale_or_none, matrix_value_scale):
            - normalized_matrix: Matrix in normalized space
            - scale_or_none: MatrixScale object for "matrix", None for "none"/"rhs"
            - matrix_value_scale: Scalar that maps stored matrix values back to raw values
              (A_raw = A_stored * matrix_value_scale). 1.0 for "none"/"rhs".

    Examples:
        >>> A_norm, scale, value_scale = normalize_matrix_for_generation(A, "matrix", None)
        >>> isinstance(scale, MatrixScale)
        True
        >>> value_scale > 0
        True
    """
    from neuralls.domain.normalization import create_scale_from_config

    # No normalization: return defensive copy
    if normalize_type in ("none", "rhs"):
        return matrix.copy(), None, 1.0

    scale = create_scale_from_config(
        normalize_type=normalize_type,
        matrix=matrix,
        spectral_radius_bound=spectral_radius_bound,
    )
    assert scale is not None, f"Expected scale for {normalize_type}"
    matrix_norm = scale.scale_matrix(matrix)
    scale_params = serialize_scale_metadata(scale)
    if scale_params is None:
        raise TypeError(f"Expected scale metadata for {normalize_type}")
    spectral_radius = scale_params.get("spectral_radius_bound")
    dimension_scale = scale_params.get("dimension_scale")
    if not isinstance(spectral_radius, float) or not isinstance(dimension_scale, float):
        raise TypeError("Matrix normalization scale metadata must be scalar floats.")
    matrix_value_scale = spectral_radius * dimension_scale
    return matrix_norm, scale, matrix_value_scale


def serialize_scale_metadata(scale: IScale | None) -> ScaleMetadata | None:
    """Serialize supported scale objects into manifest metadata."""
    if scale is None:
        return None

    payload = scale.to_dict()
    metadata: ScaleMetadata = {}

    spectral_radius = payload.get("spectral_radius_bound")
    if isinstance(spectral_radius, float):
        metadata["spectral_radius_bound"] = spectral_radius

    dimension_scale = payload.get("dimension_scale")
    if isinstance(dimension_scale, float):
        metadata["dimension_scale"] = dimension_scale

    return metadata or None


def resolve_strategy_counts(
    counts: Mapping[str, int] | None,
    mix: Mapping[str, float] | None,
    total: int | None,
) -> dict[str, int]:
    """Resolve strategy counts from explicit counts or mix/total pair.

    Pure function: no side effects.

    Args:
        counts: Optional explicit strategy counts
        mix: Optional strategy proportions
        total: Total samples (required if mix provided)

    Returns:
        Dictionary of strategy_name -> count (only positive counts)

    Raises:
        ValueError: If arguments are invalid or inconsistent
    """
    if counts is not None and mix is not None:
        raise ValueError("Specify either explicit counts or a mix/total pair, not both")

    if counts is None:
        if mix is None:
            raise ValueError("Either counts or mix must be provided")
        if total is None:
            raise ValueError("Parameter 'total' is required when using mix")
        resolved = rounded_counts(int(total), mix)
    else:
        resolved = {name: int(value) for name, value in counts.items()}

    # Filter out zero counts (but keep -1 which means "all available")
    nonzero = {name: value for name, value in resolved.items() if value != 0}
    if not nonzero:
        raise ValueError("No strategy counts were provided")

    return nonzero


@dataclass(frozen=True, slots=True)
class _SolveOptions:
    """Per-call solver settings shared by every registry entry."""

    rtol: float
    atol: float
    max_iters: int
    assume_pos_def: bool


type _SolveFn = Callable[[MatrixOperator, np.ndarray, _SolveOptions], np.ndarray]
"""Solver over a stack of RHS rows, shape (num_systems, n)."""


def _solve_direct_rows(
    operator: MatrixOperator, rhs_array: np.ndarray, options: _SolveOptions
) -> np.ndarray:
    """Direct solve per RHS row, reusing the operator's cached factor."""
    solutions = np.zeros(rhs_array.shape, dtype=np.float64)
    for idx, rhs in enumerate(rhs_array):
        solutions[idx] = operator.solve_direct(rhs, assume_pos_def=options.assume_pos_def)
    return solutions


def _solve_cg_rows(
    operator: MatrixOperator, rhs_array: np.ndarray, options: _SolveOptions
) -> np.ndarray:
    """Iterative CG per RHS row; warns on non-convergence."""
    from scipy.sparse.linalg import cg as scipy_cg

    solutions = np.zeros(rhs_array.shape, dtype=np.float64)
    for idx, rhs in enumerate(rhs_array):
        solution, exit_code = scipy_cg(
            operator.matrix,
            rhs,
            rtol=options.rtol,
            atol=options.atol,
            maxiter=options.max_iters,
        )
        solutions[idx] = solution

        if exit_code != 0:
            residual = np.linalg.norm(operator.matvec(solution) - rhs)
            logger.warning(f"System {idx + 1} CG exit_code={exit_code} (residual: {residual:.2e})")
    return solutions


type _SolveMethod = Literal["direct", "cg"]
"""Closed set of linear-solve strategies accepted by `_solve_linear_systems`."""

_SOLVERS: Mapping[_SolveMethod, _SolveFn] = MappingProxyType(
    {
        "direct": _solve_direct_rows,
        "cg": _solve_cg_rows,
    }
)


def _solve_linear_systems(
    A: MatrixOperator,
    rhs_vectors: np.ndarray,
    method: _SolveMethod,
    rtol: float = 1e-12,
    atol: float = 0.0,
    max_iters: int = 500,
    assume_pos_def: bool = True,
) -> np.ndarray:
    """Solve linear systems Ax = b using configured method and matrix format.

    The solver is looked up in `_SOLVERS` by method. Direct solves use the
    operator's cached factorization; CG works on either format.

    Args:
        A: Operator wrapping the system matrix, shape (n, n)
        rhs_vectors: RHS vectors, shape (num_systems, n)
        method: Solving method ("direct" or "cg")
        rtol: Relative tolerance for CG (ignored for direct)
        atol: Absolute tolerance for CG (ignored for direct)
        max_iters: Maximum CG iterations (ignored for direct)
        assume_pos_def: Dense direct solve only: Cholesky (True) or LU (False)

    Returns:
        Solution vectors, shape (num_systems, n)

    Raises:
        ValueError: If no solver is registered for `method`
    """
    rhs_array = np.asarray(rhs_vectors, dtype=np.float64)
    solver = _SOLVERS.get(method)
    if solver is None:
        raise ValueError(f"Invalid solve method: {method}. Must be 'direct' or 'cg'")
    options = _SolveOptions(
        rtol=rtol,
        atol=atol,
        max_iters=max_iters,
        assume_pos_def=assume_pos_def,
    )
    return solver(A, rhs_array, options)


def _build_trace_indices(
    sample_idx: int,
    indices: range,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sample and iteration index arrays for trace data.

    Pure function. `indices` is an already-resolved set of trajectory step
    indices (e.g. from `StepWindow.select_with_indices`), not a `StepWindow`
    — this keeps the "which rows were selected" and "what are their
    original indices" pairing structurally impossible to compute from two
    different arrays by mistake (see `step_window.py`'s
    `select_with_indices` docstring).

    Args:
        sample_idx: Sample index for this trace.
        indices: The trajectory step indices that were kept.

    Returns:
        Tuple of (sample_indices, iteration_indices).
    """
    iteration_indices = np.fromiter(indices, dtype=np.int64)
    return (
        np.full(iteration_indices.shape[0], sample_idx, dtype=np.int64),
        iteration_indices,
    )


def trace_rows_per_system(window: StepWindow) -> int:
    """Return the worst-case number of kept trace rows for one base system.

    Computed against `window.stop + 1` — the trajectory length when the
    safety cap is fully used, which is always the case for trajectories:
    the solver runs exactly `window.stop` steps, so every base system yields
    exactly this many rows. Used only to budget how many base systems to run up front — see
    `resolve_trace_generation_counts`.
    """
    return len(window.resolve_indices(window.stop + 1))


class _WindowedConfig(Protocol):
    """A validated strategy config that exposes its trajectory window."""

    @property
    def window(self) -> StepWindow: ...


def trace_rows_per_base_system(
    config_type: Callable[..., _WindowedConfig],
) -> Callable[[Mapping[str, Any]], int]:
    """Build the K_rows callable for a trace strategy from its config model."""

    def _rows(cfg: Mapping[str, Any]) -> int:
        return trace_rows_per_system(config_type(**cfg).window)

    return _rows


def required_trace_systems(samples: int, *, window: StepWindow) -> int:
    """Return the base-system count for a desired trace-row budget."""
    rows_per_system = trace_rows_per_system(window)
    return max(1, math.ceil(samples / rows_per_system))


def resolve_trace_generation_counts(
    samples: int,
    *,
    window: StepWindow,
    available_systems: int | None,
    strategy_name: str,
) -> tuple[int, int | None]:
    """Resolve base-system count for trajectory-harvesting strategies."""
    if samples == -1:
        if available_systems is None:
            raise ValueError(
                f"Strategy '{strategy_name}' does not support samples=-1 without "
                "a finite archive-backed source."
            )
        return available_systems, None
    return required_trace_systems(samples, window=window), samples


def _compute_eigendecomposition(
    operator: MatrixOperator,
    count: int,
    which: EigenvectorSelection,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenpairs needed to select `count` eigenvectors.

    "smallest" and "largest" ask the operator for only `count` pairs, which
    is the only option for CSR. "random" needs the whole spectrum, so it is
    dense-only; the dense full spectrum is requested as `count = n`.

    Args:
        operator: Operator wrapping the symmetric system matrix
        count: Number of eigenpairs to return for "smallest"/"largest"
        which: Selection mode

    Returns:
        Tuple of (eigenvalues ascending, eigenvectors as columns)

    Raises:
        ValueError: If the matrix is not symmetric, or "random" is requested
            for a CSR operator
    """
    match which:
        case EigenvectorSelection.RANDOM:
            if operator.format is MatrixFormat.CSR:
                raise ValueError(
                    "Random eigenvector selection needs the full spectrum, which requires "
                    "a dense matrix; use 'smallest' or 'largest' with csr matrices"
                )
            n = operator.shape[0]
            return operator.eigensystem(n, EigenvectorSelection.SMALLEST)
        case EigenvectorSelection.SMALLEST | EigenvectorSelection.LARGEST:
            return operator.eigensystem(count, which)


def _select_eigenvectors(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    count: int,
    which: EigenvectorSelection,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select subset of eigenvectors according to which eigenvalues to use.

    Args:
        eigenvectors: Eigenvector matrix, shape (n, n)
        eigenvalues: Eigenvalue array, shape (n,)
        count: Number of eigenvectors to select
        which: Which eigenvalues to select ("smallest", "largest", or "random")
            - "smallest": Select k eigenvectors with smallest eigenvalues
            - "largest": Select k eigenvectors with largest eigenvalues
            - "random": Random selection without replacement
        rng: Random number generator (used for "random" mode)

    Returns:
        Tuple of (selected_eigenvectors, selected_eigenvalues, indices)

    Raises:
        ValueError: If count is outside 1..n.
    """
    n = eigenvectors.shape[0]
    available = eigenvalues.shape[0]
    require_eigen_count(count, n)
    match which:
        case EigenvectorSelection.SMALLEST:
            indices = np.arange(count)
        case EigenvectorSelection.LARGEST:
            indices = np.arange(available - count, available)
        case EigenvectorSelection.RANDOM:
            indices = rng.choice(available, size=count, replace=False)
    return eigenvectors[:, indices], eigenvalues[indices], indices


def _generate_eigenvector_combinations(
    eigenvectors: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random L2-normalized linear combinations of eigenvectors.

    Args:
        eigenvectors: Eigenvector matrix, shape (n, k)
        num_samples: Number of combinations to generate
        rng: Random number generator

    Returns:
        Linear combinations, shape (num_samples, n)
    """
    _n, k = eigenvectors.shape
    coeffs = rng.standard_normal(size=(num_samples, k), dtype=np.float64)
    norms = np.linalg.norm(coeffs, axis=1, keepdims=True)
    coeffs_normalized = coeffs / norms
    return coeffs_normalized @ eigenvectors.T


def _verify_solution_accuracy(
    A: np.ndarray,
    rhs_vectors: np.ndarray,
    solutions: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Verify solution accuracy by computing relative residuals.

    Args:
        A: System matrix
        rhs_vectors: Right-hand side vectors
        solutions: Solution vectors
        tolerance: Relative residual tolerance for warnings

    Returns:
        Array of relative residuals, shape (count,)

    Warns:
        RuntimeWarning: If any solution exceeds tolerance
    """
    count = rhs_vectors.shape[0]
    rel_residuals = np.zeros(count)
    for i in range(count):
        residual = A @ solutions[i] - rhs_vectors[i]
        rhs_norm = np.linalg.norm(rhs_vectors[i])
        rel_residuals[i] = (
            np.linalg.norm(residual) / rhs_norm if rhs_norm > 0 else np.linalg.norm(residual)
        )
        if rel_residuals[i] > tolerance:
            warnings.warn(
                f"Sample {i}: Solution accuracy {rel_residuals[i]:.2e} "
                f"exceeds tolerance {tolerance:.2e}",
                RuntimeWarning,
                stacklevel=2,
            )
    return rel_residuals


def select_archive_files(
    glob_pattern: str,
    count: int,
    shuffle: bool,
    seed: int | None,
    skip: int = 0,
    file_indices: tuple[int, ...] | None = None,
) -> list:
    """Select N files from glob pattern with optional shuffling.

    Pure function for deterministic file selection from archives.

    Args:
        glob_pattern: Path pattern like "/data/*.txt" or "/data/rhs_*.npy"
        count: Number of files to select (-1 means all available files)
        shuffle: Whether to shuffle before selection
        seed: Random seed for shuffling (required if shuffle=True)
        skip: Number of files to skip after deterministic ordering/shuffling
        file_indices: Explicit positions in the pool that remains after ``skip``. When
            given, ``count`` must equal their number and exactly these files are returned,
            in this order. Lets the caller give each binding a disjoint slice of one pool.

    Returns:
        List of selected file paths (pathlib.Path objects)

    Raises:
        FileNotFoundError: If no files match pattern or directory doesn't exist
        ValueError: If count and skip exceed available files

    Examples:
        >>> # Select first 10 files
        >>> files = select_archive_files("/data/rhs_*.txt", 10, False, None)

        >>> # Select all files with shuffling
        >>> files = select_archive_files("/data/sol_*.npy", -1, True, 42)

        >>> # Select 50 shuffled files
        >>> files = select_archive_files("/data/vec_*.txt", 50, True, 42)
    """
    pattern_path = Path(glob_pattern)
    directory = pattern_path.parent
    pattern = pattern_path.name

    # Validate directory exists
    if not directory.exists():
        raise FileNotFoundError(f"Archive directory not found: {directory}")

    # Scan for matching files
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files found matching pattern: {directory / pattern}")

    if skip < 0:
        raise ValueError(f"Archive skip must be non-negative, got {skip}")
    if skip > len(candidates):
        raise ValueError(
            f"Requested skip={skip} but only {len(candidates)} files are available "
            f"matching pattern: {glob_pattern}"
        )

    ordered = _ordered_candidates(candidates, shuffle, seed)
    if file_indices is not None:
        return _pick_explicit_files(ordered[skip:], count, file_indices, glob_pattern)

    # Handle "all files" case
    if count == -1:
        count = len(candidates) - skip

    # Validate sufficient files available
    if skip + count > len(candidates):
        if skip == 0:
            raise ValueError(
                f"Requested {count} files but only {len(candidates)} available "
                f"matching pattern: {glob_pattern}"
            )
        raise ValueError(
            f"Requested {count} files with skip={skip} but only {len(candidates)} available "
            f"matching pattern: {glob_pattern}"
        )

    return ordered[skip : skip + count]


def _ordered_candidates(candidates: list, shuffle: bool, seed: int | None) -> list:
    """Return the sorted candidates, permuted by ``seed`` when ``shuffle`` is set."""
    if not shuffle:
        return candidates
    rng = rng_from_seed(seed)
    return [candidates[idx] for idx in rng.permutation(len(candidates))]


def solution_row_count(path: Path) -> int:
    """Count the rows of an explicit solution file without reading its values.

    .npy rows come from the array header; a 1-D array is one sample, matching the
    vector reader. A .txt file is one sample whatever its line count: the vector
    reader loads the whole file with ``np.loadtxt`` as a single vector, so counting
    lines would promise rows the reader never yields. Other formats have no cheap
    row count here and are rejected by name.

    Raises:
        ValueError: If the format has no cheap row count, or the .npy rank is not 1 or 2.
    """
    match path.suffix:
        case ".npy":
            with path.open("rb") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"Unsupported .npy version {version} in solution file {path}.")
            match len(shape):
                case 1:
                    return 1
                case 2:
                    return int(shape[0])
                case _:
                    raise ValueError(
                        f"Solution .npy file {path} must have shape (n,) or (N,n), got {shape}."
                    )
        case ".txt":
            return 1
        case _:
            raise ValueError(
                f"Cannot read the row count of solution file {path} (format '{path.suffix}'): "
                "only .npy and .txt are supported."
            )


def _pick_explicit_files(
    pool: list,
    count: int,
    file_indices: tuple[int, ...],
    glob_pattern: str,
) -> list:
    """Return the pool entries at ``file_indices`` after checking they form a valid selection."""
    if count != len(file_indices):
        raise ValueError(
            f"count={count} must equal the {len(file_indices)} explicit file indices "
            f"matching pattern: {glob_pattern}"
        )
    out_of_range = [idx for idx in file_indices if not 0 <= idx < len(pool)]
    if out_of_range:
        raise ValueError(
            f"Explicit file indices {out_of_range} are outside the {len(pool)} files "
            f"available matching pattern: {glob_pattern}"
        )
    return [pool[idx] for idx in file_indices]


# =============================================================================
# KRYLOV STRATEGY HELPERS
# =============================================================================


def _lanczos_iteration(
    matrix: np.ndarray,
    krylov_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Lanczos iteration to build Krylov subspace basis.

    Args:
        matrix: System matrix A
        krylov_dim: Dimension of Krylov subspace
        rng: Random number generator

    Returns:
        Tuple of (V, T) where:
            - V: Orthonormal basis vectors, shape (n, m_eff)
            - T: Tridiagonal matrix, shape (m_eff, m_eff)
            m_eff <= krylov_dim (early termination if breakdown occurs)
    """
    n = matrix.shape[0]
    m = krylov_dim
    V = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m + 1, dtype=np.float64)

    # Initial random vector
    v = rng.normal(size=n).astype(np.float64, copy=False)
    v = v / norm(v)
    V[:, 0] = v

    v_prev = np.zeros(n, dtype=np.float64)
    beta[0] = 0.0

    # Lanczos iteration
    m_eff = m
    for j in range(m):
        w = matrix @ V[:, j] - beta[j] * v_prev
        alpha[j] = np.dot(V[:, j], w)
        w = w - alpha[j] * V[:, j]
        beta[j + 1] = norm(w)

        # Check for breakdown (lucky breakdown)
        if beta[j + 1] <= 1e-14:
            m_eff = j + 1
            V = V[:, :m_eff]
            alpha = alpha[:m_eff]
            beta = beta[: m_eff + 1]
            break

        v_prev = V[:, j].copy()
        if j + 1 < m:
            V[:, j + 1] = w / beta[j + 1]

    # Build tridiagonal matrix
    T = np.diag(alpha[:m_eff]) + np.diag(beta[1:m_eff], k=-1) + np.diag(beta[1:m_eff], k=1)

    return V, T


def _generate_krylov_combinations(
    V: np.ndarray,
    T: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random linear combinations from Krylov basis.

    Args:
        V: Orthonormal Krylov basis, shape (n, m_eff)
        T: Tridiagonal matrix from Lanczos, shape (m_eff, m_eff)
        num_samples: Number of combinations to generate
        rng: Random number generator

    Returns:
        Linear combinations, shape (num_samples, n)
    """
    m_eff = V.shape[1]

    # Eigendecomposition of tridiagonal matrix
    Lambda, Q = np.linalg.eigh(T)
    Lambda_inv = 1.0 / Lambda

    # Generate random combinations
    combinations = []
    for _ in range(num_samples):
        eps = rng.normal(size=m_eff).astype(np.float64, copy=False)
        x = V @ (Q @ (Lambda_inv * eps))
        combinations.append(x)

    return np.array(combinations, dtype=np.float64)


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
    "normalize_matrix_for_generation",
    "resolve_strategy_counts",
    "rng_from_seed",
    "rounded_counts",
    "select_archive_files",
]
