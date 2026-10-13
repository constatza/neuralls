"""Tests for ``runner.py::_realized_row_count``.

Regression coverage for a found bug: ``smoother_filtered_probes`` returns a
``GeneratedSamples`` with ``rhs``/``solutions`` both ``None`` (only
``residual_traces`` populated) — falling through to ``matrix.shape[0]``
there logged the system dimension instead of the realized sample count.
"""

from __future__ import annotations

import numpy as np

from neuralls.domain.generation.interfaces import GeneratedSamples
from neuralls.domain.generation.runner import _realized_row_count
from neuralls.domain.normalization import ErrorTraceSamples, ResidualTraceSamples


def _matrix(n: int = 4) -> np.ndarray:
    return np.eye(n)


def test_prefers_rhs_row_count() -> None:
    """When `rhs` is populated, its row count wins."""
    result = GeneratedSamples(
        matrix=_matrix(50),
        rhs=np.zeros((7, 50)),
        solutions=np.zeros((7, 50)),
    )
    assert _realized_row_count(result) == 7


def test_falls_back_to_solutions_when_rhs_is_none() -> None:
    """When only `solutions` is populated, its row count is used."""
    result = GeneratedSamples(matrix=_matrix(50), rhs=None, solutions=np.zeros((9, 50)))
    assert _realized_row_count(result) == 9


def test_uses_residual_traces_row_count_not_matrix_dimension() -> None:
    """A residual-trace-only result (e.g. smoother_filtered_probes) reports its real row count.

    The matrix is 504x504 but only 11 trace rows were realized — the
    regression this guards against logged 504 ("N=504") instead of 11.
    """
    traces = ResidualTraceSamples(
        residuals=np.zeros((11, 504)),
        solutions=np.zeros((11, 504)),
        sample_indices=np.zeros(11, dtype=int),
        iteration_indices=np.zeros(11, dtype=int),
    )
    result = GeneratedSamples(matrix=_matrix(504), rhs=None, solutions=None, residual_traces=traces)
    assert _realized_row_count(result) == 11


def test_uses_error_traces_row_count_not_matrix_dimension() -> None:
    """An error-trace-only result reports its real row count, not the matrix dimension."""
    traces = ErrorTraceSamples(
        residuals=np.zeros((5, 30)),
        solutions_current=np.zeros((5, 30)),
        errors=np.zeros((5, 30)),
        true_solutions=np.zeros((5, 30)),
        sample_indices=np.zeros(5, dtype=int),
        iteration_indices=np.zeros(5, dtype=int),
    )
    result = GeneratedSamples(matrix=_matrix(30), rhs=None, solutions=None, error_traces=traces)
    assert _realized_row_count(result) == 5


def test_falls_back_to_matrix_dimension_when_nothing_else_is_set() -> None:
    """With no row-shaped field populated at all, the matrix dimension is the only option left."""
    result = GeneratedSamples(matrix=_matrix(13), rhs=None, solutions=None)
    assert _realized_row_count(result) == 13
