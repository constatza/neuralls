"""Format-aware wrapper around one system matrix A.

Owns the expensive per-matrix work (LU/Cholesky factorization, eigen
decomposition) and memoizes it, so that the many samples generated from one
matrix share a single factorization.

Pure linear algebra: no file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, lu_factor, lu_solve
from scipy.sparse import csr_array
from scipy.sparse.linalg import SuperLU, eigsh, splu

from neuralls.shared.types import EigenvectorSelection, MatrixFormat, SystemMatrix

type EigenWhich = Literal[EigenvectorSelection.SMALLEST, EigenvectorSelection.LARGEST]
"""Eigenvalue end requested from `MatrixOperator.eigensystem`."""

SYMMETRY_ATOL = 1e-10
"""Largest allowed |A - A^T| entry, the same absolute rule for dense and CSR matrices.

Generated systems are normalized to entries of order one, so an absolute bound is
the right scale. A relative bound would let large entries hide a larger asymmetry.
"""

_SHIFT_INVERT_SIGMA = 0.0
"""Shift for `eigsh` shift-invert when computing the smallest eigenvalues.

The shift at zero targets the smallest eigenvalues of an SPD matrix, which
are the ones closest to the shift. It is used with which="LM" because
`eigsh` only supports magnitude selection under shift-invert.
"""


class _FactorKind(StrEnum):
    """Which factorization is cached on an operator."""

    SPLU = "splu"
    CHOLESKY = "cholesky"
    LU = "lu"


@dataclass(frozen=True)
class _SparseLUFactor:
    """Sparse LU factorization of a CSR matrix."""

    kind: ClassVar[_FactorKind] = _FactorKind.SPLU
    lu: SuperLU

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve A x = rhs with the cached factor."""
        return np.asarray(self.lu.solve(rhs), dtype=np.float64)


@dataclass(frozen=True)
class _CholeskyFactor:
    """Cholesky factorization of a dense SPD matrix, as returned by `cho_factor`."""

    kind: ClassVar[_FactorKind] = _FactorKind.CHOLESKY
    factor: tuple[np.ndarray, bool]

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve A x = rhs with the cached factor."""
        return np.asarray(cho_solve(self.factor, rhs), dtype=np.float64)


@dataclass(frozen=True)
class _LUFactor:
    """LU factorization of a dense general matrix, as returned by `lu_factor`."""

    kind: ClassVar[_FactorKind] = _FactorKind.LU
    factor: tuple[np.ndarray, np.ndarray]

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve A x = rhs with the cached factor."""
        return np.asarray(lu_solve(self.factor, rhs), dtype=np.float64)


type _Factor = _SparseLUFactor | _CholeskyFactor | _LUFactor


@dataclass(slots=True)
class _OperatorCache:
    """Mutable memo for one `MatrixOperator`; excluded from value identity."""

    factor: _Factor | None = None
    eigen: dict[tuple[int, EigenWhich], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def _max_asymmetry(matrix: SystemMatrix) -> float:
    """Largest absolute entry of A - A^T, computed without densifying CSR."""
    if isinstance(matrix, csr_array):
        difference = abs(matrix - matrix.T)
        return float(difference.max()) if difference.nnz else 0.0
    return float(np.max(np.abs(matrix - matrix.T)))


def require_eigen_count(count: int, n: int) -> None:
    """Reject an eigenpair count outside 1..n. Every eigen path shares this rule.

    Raises:
        ValueError: If ``count`` is not in 1..n.
    """
    if not 0 < count <= n:
        raise ValueError(f"count ({count}) must be positive and ≤ {n}")


def ensure_symmetric(matrix: SystemMatrix) -> None:
    """Raise if `matrix` is not symmetric within `SYMMETRY_ATOL`.

    Eigenvector strategies rely on an orthogonal eigenbasis, which only exists
    for symmetric A, so asymmetry is rejected before any eigen computation.

    Raises:
        ValueError: If the maximum absolute asymmetry exceeds `SYMMETRY_ATOL`.
    """
    max_asymmetry = _max_asymmetry(matrix)
    if max_asymmetry > SYMMETRY_ATOL:
        raise ValueError(
            f"Eigenvector strategies require symmetric matrices. Max asymmetry: {max_asymmetry:.2e}"
        )


def _factor_kind(matrix_format: MatrixFormat, assume_pos_def: bool) -> _FactorKind:
    """Map a format and SPD assumption to the factorization that serves it."""
    match matrix_format:
        case MatrixFormat.CSR:
            return _FactorKind.SPLU
        case MatrixFormat.DENSE:
            return _FactorKind.CHOLESKY if assume_pos_def else _FactorKind.LU


@dataclass(frozen=True)
class MatrixOperator:
    """Immutable view of one system matrix with cached factorizations.

    The factorization and eigen results are computed lazily and memoized on a
    private cache, so repeated calls on one operator do not refactor A.

    Attributes:
        matrix: The wrapped matrix, dense ndarray or CSR array. Square 2-D.
    """

    matrix: SystemMatrix
    _cache: _OperatorCache = field(default_factory=_OperatorCache, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Reject unsupported types and non-square matrices.

        Raises:
            TypeError: If `matrix` is not a dense ndarray or a CSR array
                (this includes scipy `LinearOperator`, which has no factorization).
            ValueError: If `matrix` is not square 2-D.
        """
        if not isinstance(self.matrix, np.ndarray | csr_array):
            raise TypeError(
                "MatrixOperator requires a dense ndarray or csr_array, "
                f"got {type(self.matrix).__name__}."
            )
        rows, cols = self.matrix.shape
        if rows != cols:
            raise ValueError(f"System matrix must be square, got shape {self.matrix.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape (n, n)."""
        return self.matrix.shape

    @property
    def format(self) -> MatrixFormat:
        """Storage format of the wrapped matrix."""
        return MatrixFormat.CSR if isinstance(self.matrix, csr_array) else MatrixFormat.DENSE

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Return A @ x without densifying a CSR matrix."""
        return np.asarray(self.matrix @ x, dtype=np.float64)

    def solve_direct(self, rhs: np.ndarray, *, assume_pos_def: bool = True) -> np.ndarray:
        """Solve A x = rhs using the cached factorization.

        The factor is built on the first call and reused afterwards. Only the
        DENSE format depends on `assume_pos_def`: True selects a Cholesky
        factor (`assume_a="pos"`), False an LU factor (`assume_a="gen"`).
        The CSR path always uses sparse LU and ignores the flag. Changing the
        flag between calls rebuilds the dense factor.

        Args:
            rhs: Right-hand side, shape (n,).
            assume_pos_def: Dense-only choice between Cholesky and LU.

        Returns:
            Solution vector, shape (n,).
        """
        kind = _factor_kind(self.format, assume_pos_def)
        if self._cache.factor is None or self._cache.factor.kind is not kind:
            self._cache.factor = self._factorize(kind)
        return self._cache.factor.solve(rhs)

    def _factorize(self, kind: _FactorKind) -> _Factor:
        """Build the factorization for `kind` from the wrapped matrix."""
        match kind:
            case _FactorKind.SPLU:
                return _SparseLUFactor(splu(csr_array(self.matrix).tocsc()))
            case _FactorKind.CHOLESKY:
                return _CholeskyFactor(cho_factor(np.asarray(self.matrix, dtype=np.float64)))
            case _FactorKind.LU:
                return _LUFactor(lu_factor(np.asarray(self.matrix, dtype=np.float64)))

    def eigensystem(self, count: int, which: EigenWhich) -> tuple[np.ndarray, np.ndarray]:
        """Return `count` eigenpairs at one end of the spectrum, ascending.

        DENSE uses a full `eigh` and slices. CSR uses `eigsh`: smallest uses
        shift-invert at `sigma=0` with which="LM" (the operator's LU factor
        cannot be passed into `eigsh`, so a separate factorization is made
        inside ARPACK), largest uses which="LA". Results are cached per
        (count, which).

        Args:
            count: Number of eigenpairs to return.
            which: "smallest" or "largest" eigenvalues.

        Returns:
            Tuple of (eigenvalues ascending, eigenvectors as columns (n, count)).

        Raises:
            ValueError: If the matrix is not symmetric, `count` is out of range,
                or a CSR request would need `count >= n` (ARPACK limit).
        """
        n = self.shape[0]
        require_eigen_count(count, n)
        key = (count, which)
        if key in self._cache.eigen:
            return self._cache.eigen[key]
        ensure_symmetric(self.matrix)
        result = self._compute_eigensystem(count, which, n)
        self._cache.eigen[key] = result
        return result

    def _compute_eigensystem(
        self, count: int, which: EigenWhich, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        match self.format:
            case MatrixFormat.DENSE:
                values, vectors = eigh(np.asarray(self.matrix, dtype=np.float64))
                return _take_end(values, vectors, count, which)
            case MatrixFormat.CSR:
                if count >= n:
                    raise ValueError(
                        f"Sparse eigensolve needs count < n (got {count} for n={n}); "
                        "use a dense matrix for the full spectrum"
                    )
                csr = csr_array(self.matrix)
                if which == EigenvectorSelection.SMALLEST:
                    values, vectors = eigsh(csr, k=count, sigma=_SHIFT_INVERT_SIGMA, which="LM")
                else:
                    values, vectors = eigsh(csr, k=count, which="LA")
                order = np.argsort(values)
                return values[order], vectors[:, order]


def _take_end(
    values: np.ndarray, vectors: np.ndarray, count: int, which: EigenWhich
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the `count` ascending eigenpairs at the requested end."""
    if which == EigenvectorSelection.SMALLEST:
        return values[:count], vectors[:, :count]
    return values[-count:], vectors[:, -count:]
