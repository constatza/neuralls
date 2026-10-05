"""Solution validation utilities for iterative solvers.

This module provides reusable validation functions for checking solution
validity and recording breakdown events. These utilities are shared across
all solver implementations.

Design:
    - Pure functions (no state)
    - Reusable across solver types
    - Consistent breakdown event recording
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_array

from neuralls.shared.types import SystemMatrix


def validate_matrix(A: SystemMatrix) -> None:
    """Validate system matrix is square, 2-D, and finite.

    For CSR matrices only the stored values (``A.data``) are checked. Implicit
    zeros are never stored, so they are finite by construction; explicit
    zeros are finite too and need no separate check.

    Args:
        A: Input matrix, dense ``ndarray`` or ``csr_array``.

    Raises:
        ValueError: If matrix is not 2-D, not square, or contains non-finite values.
    """
    if A.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got {A.ndim}D")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")
    values = A.data if isinstance(A, csr_array) else A
    if not np.isfinite(values).all():
        raise ValueError("Matrix contains non-finite values")


def validate_rhs_vector(b: np.ndarray | None, A: SystemMatrix | None = None) -> None:
    """Validate RHS vector dimensions and finiteness.

    Args:
        b: RHS vector.
        A: Optional system matrix for size checking.

    Raises:
        ValueError: If RHS is invalid.
    """
    if b is None:
        return
    if b.ndim > 2:
        raise ValueError(f"RHS must be 1D or 2D, got {b.ndim}D")
    if b.ndim == 2 and b.shape[1] != 1:
        raise ValueError(f"RHS must be a column vector, got shape {b.shape}")
    if not np.isfinite(b).all():
        raise ValueError("RHS contains non-finite values")
    if A is not None and len(b.flatten()) != A.shape[0]:
        raise ValueError(f"RHS length {len(b.flatten())} doesn't match matrix size {A.shape[0]}")


def validate_ax_equals_b(matrix: SystemMatrix, rhs: np.ndarray, lhs: np.ndarray) -> None:
    """Verify ``matrix @ lhs`` and ``rhs`` are consistent up to a positive scalar.

    Scale-agnostic (doesn't assume any particular multiplier baked into rhs):
    checks the two vectors are parallel via cosine similarity. Intended as a
    direct runtime proof that a loaded/normalized linear system is internally
    consistent whenever a known true solution (``lhs``) is available — a
    structural safety net against double-normalization or mismatched-scale
    bugs, not just careful-by-construction code.

    Args:
        matrix: System matrix A.
        rhs: Right-hand side vector b.
        lhs: Known true solution x.

    Raises:
        ValueError: If the Ax=b invariant is violated beyond floating tolerance.
    """
    predicted = matrix @ lhs
    predicted_norm = float(np.linalg.norm(predicted))
    rhs_norm = float(np.linalg.norm(rhs))
    if predicted_norm == 0.0 or rhs_norm == 0.0:
        return
    cosine = float(np.dot(predicted, rhs) / (predicted_norm * rhs_norm))
    if cosine < 1.0 - 1e-6:
        raise ValueError(
            "Ax=b invariant violated: cosine similarity between (matrix @ lhs) "
            f"and rhs is {cosine:.10f} (expected ~1.0). This indicates a "
            "double-normalization or mismatched-scale bug in the linear system."
        )


if TYPE_CHECKING:
    from numpy.typing import NDArray


def check_solution_validity(solution: NDArray) -> bool:
    """Check if solution contains only finite values.

    Args:
        solution: Solution vector to validate

    Returns:
        True if solution is valid (all finite), False otherwise

    Example:
        >>> import numpy as np
        >>> check_solution_validity(np.array([1.0, 2.0, 3.0]))
        True
        >>> check_solution_validity(np.array([1.0, np.nan, 3.0]))
        False
        >>> check_solution_validity(np.array([1.0, np.inf, 3.0]))
        False
    """
    return bool(np.all(np.isfinite(solution)))


def record_breakdown_event(
    event_log: Any | None,
    solution: NDArray,
    iteration: int,
) -> None:
    """Record breakdown event when solution contains NaN/Inf.

    Args:
        event_log: EventLog to record to (can be None)
        solution: Solution vector with invalid values
        iteration: Final iteration number

    Example:
        >>> import numpy as np
        >>> class EventLog:
        ...     def record(self, event_type: str, *, iteration: int, reason: str) -> None:
        ...         self.payload = (event_type, iteration, reason)
        >>> event_log = EventLog()
        >>> record_breakdown_event(event_log, np.array([1.0, np.nan]), iteration=5)
    """
    if event_log is None:
        return

    has_nan = bool(np.any(np.isnan(solution)))
    has_inf = bool(np.any(np.isinf(solution)))

    reason_parts = []
    if has_nan:
        reason_parts.append("nan_in_solution")
    if has_inf:
        reason_parts.append("inf_in_solution")

    event_log.record(
        "breakdown",
        iteration=iteration,
        reason=", ".join(reason_parts),
    )
