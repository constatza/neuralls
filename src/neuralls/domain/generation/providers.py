"""Input providers for generation strategies (SOLID Layer 1: Data Provision).

Providers abstract the source of input data (solutions, RHS, or both) from the
transformation logic. This enables strategies to be agnostic to whether data comes
from random generation, archives, files, or external sources.

Architecture:
    - InputProvider: Protocol for any data source
    - Concrete implementations: Random, File, Constant, Hybrid, etc.
    - Strategies compose providers to obtain inputs

Examples:
    >>> # Random input
    >>> provider = RandomInputProvider(seed=42, scale=1.0)
    >>> solutions = provider.provide(matrix, count=50, rng=rng)

    >>> # File input
    >>> provider = FileInputProvider(glob_pattern="/data/sols_*.npy")
    >>> solutions = provider.provide(matrix, count=50, rng=rng)

    >>> # Hybrid (archive with fallback)
    >>> provider = HybridInputProvider(archive=archive_data)
    >>> solutions = provider.provide(matrix, count=50, rng=rng)
"""

from __future__ import annotations

from abc import abstractmethod
from functools import lru_cache
from typing import Protocol, TypeVar

import numpy as np

from .helpers import select_archive_files
from .interfaces import ArchiveData, ArchiveField

T_co = TypeVar("T_co", covariant=True)


class InputProvider(Protocol[T_co]):
    """Protocol for input data providers.

    Providers abstract the source of input data, enabling strategies to be
    agnostic to whether data comes from random generation, archives, files,
    or external sources.
    """

    @abstractmethod
    def provide(
        self,
        matrix: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> T_co:
        """Provide input data.

        Args:
            matrix: System matrix (may be used for dimension inference)
            count: Number of vectors requested
            rng: Random number generator (for stochastic providers)

        Returns:
            Input data of type T (e.g., np.ndarray for solutions/RHS)
        """
        ...


class RandomInputProvider:
    """Generate random normal vectors.

    Examples:
        >>> provider = RandomInputProvider(seed=42, scale=1.0)
        >>> solutions = provider.provide(matrix=np.eye(10), count=5, rng=rng)
        >>> solutions.shape
        (5, 10)
    """

    def __init__(self, seed: int | None = None, scale: float = 1.0) -> None:
        """Initialize random provider.

        Args:
            seed: Random seed for reproducibility
            scale: Standard deviation for normal distribution
        """
        self.seed = seed
        self.scale = scale

    def provide(
        self,
        matrix: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate random normal vectors.

        Args:
            matrix: System matrix (used for dimension)
            count: Number of vectors
            rng: Random number generator (ignored if seed provided)

        Returns:
            Random vectors, shape (count, n)
        """
        n = matrix.shape[0]
        # Use local RNG if seed provided, otherwise use passed RNG
        local_rng = np.random.default_rng(self.seed) if self.seed is not None else rng
        return local_rng.normal(size=(count, n), scale=self.scale).astype(np.float64, copy=False)


class ConstantInputProvider:
    """Generate constant vectors (e.g., ones, zeros).

    Examples:
        >>> provider = ConstantInputProvider(value=1.0)
        >>> solutions = provider.provide(matrix=np.eye(10), count=5, rng=rng)
        >>> np.all(solutions == 1.0)
        True
    """

    def __init__(self, value: float = 1.0) -> None:
        """Initialize constant provider.

        Args:
            value: Constant value for all entries
        """
        self.value = value

    def provide(
        self,
        matrix: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate constant vectors.

        Args:
            matrix: System matrix (used for dimension)
            count: Number of vectors
            rng: Random number generator (unused)

        Returns:
            Constant vectors, shape (count, n)
        """
        n = matrix.shape[0]
        return np.full((count, n), self.value, dtype=np.float64)


class GaussianInputProvider:
    """Generate Gaussian-distributed vectors."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu = mu
        self.sigma = sigma

    def provide(
        self,
        matrix: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate Gaussian-distributed vectors."""
        n = matrix.shape[0]
        return rng.normal(loc=self.mu, scale=self.sigma, size=(count, n)).astype(
            np.float64, copy=False
        )


class UniformInputProvider:
    """Generate uniformly distributed vectors."""

    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        self.a = a
        self.b = b

    def provide(
        self,
        matrix: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate uniformly distributed vectors."""
        n = matrix.shape[0]
        return rng.uniform(self.a, self.b, size=(count, n)).astype(np.float64, copy=False)


@lru_cache(maxsize=128)
def _load_archive_files_cached(
    glob_pattern: str,
    count: int,
    shuffle: bool,
    seed: int | None,
    skip: int,
    file_indices: tuple[int, ...] | None,
) -> tuple[np.ndarray, ...]:
    """Load and cache solution files for a given glob/selection key.

    Args:
        glob_pattern: Pattern like "/data/sols_*.npy".
        count: Number of vectors (-1 for all available).
        shuffle: Whether to shuffle files before selection.
        seed: Random seed for shuffling.
        skip: Number of files to skip after ordering/shuffling.
        file_indices: Explicit pool positions to load, or None for the contiguous selection.
            Part of the cache key, so two bindings with different slices never share an entry.

    Returns:
        Loaded vectors as a tuple (hashable/immutable for caching).
    """
    files = select_archive_files(
        glob_pattern,
        count=count,
        shuffle=shuffle,
        seed=seed,
        skip=skip,
        file_indices=file_indices,
    )
    return tuple(np.loadtxt(f) for f in files)


class FileInputProvider:
    """Load vectors from files matching glob pattern.

    Examples:
        >>> provider = FileInputProvider(glob_pattern="/data/sols_*.npy", shuffle=True, seed=42)
        >>> solutions = provider.provide(matrix=np.eye(10), count=5, rng=rng)
        >>> solutions.shape
        (5, 10)
    """

    def __init__(
        self,
        glob_pattern: str,
        shuffle: bool = False,
        seed: int | None = None,
        skip: int = 0,
        file_indices: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize file provider.

        Args:
            glob_pattern: Pattern like "/data/sols_*.npy"
            shuffle: Whether to shuffle files before selection
            seed: Random seed for shuffling
            skip: Number of files to skip after deterministic ordering/shuffling
            file_indices: Explicit positions in the pool after ``skip``; ``count`` must
                equal their number. None selects the contiguous range instead.
        """
        self.glob_pattern = glob_pattern
        self.shuffle = shuffle
        self.seed = seed
        self.skip = skip
        self.file_indices = file_indices

    def provide(
        self,
        matrix: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Load vectors from files.

        Args:
            matrix: System matrix (unused, for protocol compliance)
            count: Number of vectors (-1 for all available)
            rng: Random number generator (unused, seed used instead)

        Returns:
            Loaded vectors, shape (count, n)

        Raises:
            FileNotFoundError: If no files match pattern
            ValueError: If insufficient files available
        """
        vectors = _load_archive_files_cached(
            self.glob_pattern, count, self.shuffle, self.seed, self.skip, self.file_indices
        )
        return np.array(vectors, dtype=np.float64)


class HybridInputProvider:
    """Hybrid provider: try archive first, fallback to random.

    This provider attempts to use archive data if available, falling back to
    random generation if archive is None or has insufficient vectors.

    Examples:
        >>> # With archive
        >>> archive = ArchiveData(lhs=np.random.randn(20, 10))
        >>> provider = HybridInputProvider(archive, field=ArchiveField.LHS)
        >>> solutions = provider.provide(matrix=np.eye(10), count=5, rng=rng)
        >>> solutions.shape
        (5, 10)

        >>> # Without archive (fallback to random)
        >>> provider = HybridInputProvider(None, field=ArchiveField.LHS, scale=1.0)
        >>> solutions = provider.provide(matrix=np.eye(10), count=5, rng=rng)
        >>> solutions.shape
        (5, 10)
    """

    def __init__(
        self,
        archive: ArchiveData | None,
        field: ArchiveField = ArchiveField.LHS,
        scale: float = 1.0,
    ) -> None:
        """Initialize hybrid provider.

        Args:
            archive: Optional archive data
            field: Field to extract from archive (ArchiveField.LHS or ArchiveField.RHS)
            scale: Scale for random fallback
        """
        self.archive = archive
        self.field = field
        self.scale = scale

    def provide(
        self,
        matrix: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Provide vectors from archive or generate random.

        Args:
            matrix: System matrix (used for dimension if generating)
            count: Number of vectors
            rng: Random number generator (for fallback)

        Returns:
            Vectors from archive or random, shape (count, n)

        Raises:
            ValueError: If archive has insufficient vectors
        """
        # Try archive first
        if self.archive is not None:
            data = getattr(self.archive, self.field, None)
            if data is not None:
                if count == -1:
                    return data.astype(np.float64, copy=True)
                if data.shape[0] < count:
                    raise ValueError(
                        f"Not enough archive {self.field}: need {count}, got {data.shape[0]}"
                    )
                return data[:count].astype(np.float64, copy=True)

        # Fallback to random
        if count == -1:
            raise ValueError(f"Cannot use count=-1 for field '{self.field}' without archive data.")
        n = matrix.shape[0]
        return rng.normal(size=(count, n), scale=self.scale).astype(np.float64, copy=False)


class PairedFileInputProvider:
    """Load paired (solutions, RHS) from files.

    Examples:
        >>> provider = PairedFileInputProvider(
        ...     solution_glob="/data/sols_*.npy", rhs_glob="/data/rhs_*.npy"
        ... )
        >>> solutions, rhs = provider.provide(matrix=np.eye(10), count=5, rng=rng)
        >>> solutions.shape, rhs.shape
        ((5, 10), (5, 10))
    """

    def __init__(
        self,
        solution_glob: str,
        rhs_glob: str,
        shuffle: bool = False,
        seed: int | None = None,
        skip: int = 0,
        file_indices: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize paired file provider.

        Args:
            solution_glob: Pattern for solution files
            rhs_glob: Pattern for RHS files
            shuffle: Whether to shuffle files
            seed: Random seed for shuffling
            skip: Number of pairs to skip after deterministic ordering/shuffling
            file_indices: Explicit pair positions after ``skip``, applied to both globs.
        """
        self.solution_provider = FileInputProvider(solution_glob, shuffle, seed, skip, file_indices)
        self.rhs_provider = FileInputProvider(rhs_glob, shuffle, seed, skip, file_indices)

    def provide(
        self,
        matrix: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load paired vectors from files.

        Args:
            matrix: System matrix (unused)
            count: Number of pairs
            rng: Random number generator (unused)

        Returns:
            Tuple of (solutions, rhs), both shape (count, n)
        """
        solutions = self.solution_provider.provide(matrix, count, rng)
        rhs = self.rhs_provider.provide(matrix, count, rng)
        return solutions, rhs


def provide_solutions(
    matrix: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    solutions_glob: str | None,
    archive: ArchiveData | None,
    shuffle: bool,
    seed: int | None,
    strategy_name: str,
    skip: int = 0,
    file_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Load solution vectors from glob, archive, or raise immediately.

    Centralises the three-way dispatch used by residual trace strategies so the
    logic is not repeated in every strategy class.

    Args:
        matrix: System matrix (used for dimension validation by providers).
        count: Number of solution vectors to return.
        rng: Random number generator for providers that need it.
        solutions_glob: Optional glob pattern. Used only when no archive is available.
        archive: Optional in-memory archive. Preferred over solutions_glob when present.
        shuffle: Whether to shuffle the loaded files (passed to FileInputProvider).
        seed: Random seed for shuffling (passed to FileInputProvider).
        strategy_name: Name used in the error message when no source is available.
        skip: Number of solution files to skip after deterministic ordering/shuffling.
        file_indices: Explicit pool positions after ``skip`` (see ``FileInputProvider``),
            or None for the contiguous selection.

    Returns:
        Solution vectors, shape (count, n).

    Raises:
        ValueError: If neither solutions_glob nor archive solutions are available.
    """
    if archive is not None and archive.lhs is not None:
        return HybridInputProvider(archive=archive, field=ArchiveField.LHS, scale=1.0).provide(
            matrix, count=count, rng=rng
        )
    if solutions_glob is not None:
        return FileInputProvider(
            solutions_glob,
            shuffle=shuffle,
            seed=seed,
            skip=skip,
            file_indices=file_indices,
        ).provide(matrix, count=count, rng=rng)
    raise ValueError(
        f"{strategy_name} requires 'solutions_glob' in config or "
        "archive solutions passed in ArchiveData."
    )


__all__ = [
    "ConstantInputProvider",
    "FileInputProvider",
    "GaussianInputProvider",
    "HybridInputProvider",
    "InputProvider",
    "PairedFileInputProvider",
    "RandomInputProvider",
    "UniformInputProvider",
    "provide_solutions",
]
