"""One strategy's generated rows for one binding, as a bounded chunk of a generation run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class SampleBatch:
    """Rows produced by one strategy on one binding.

    Every per-row array shares the leading row dimension. ``parameter_vectors`` holds one
    vector per parameter stream, shared by all rows of the binding, or ``None`` when the
    binding has no sample from that stream.

    Attributes:
        binding_index: Position of the binding in the run's binding sequence.
        strategy_name: Name of the strategy that produced the rows.
        rhs: Generated feature rows, shape (rows, n).
        solutions: Generated target rows, shape (rows, n).
        row_kind_codes: Per-row ``RowKind`` codes, shape (rows,).
        matrix_sample_index: Matrix sample id of each row, shape (rows,).
        parameter_vectors: Per parameter stream, the binding's vector of shape (p,), or None.
    """

    binding_index: int
    strategy_name: str
    rhs: np.ndarray
    solutions: np.ndarray
    row_kind_codes: np.ndarray
    matrix_sample_index: np.ndarray
    parameter_vectors: tuple[np.ndarray | None, ...]

    def __post_init__(self) -> None:
        """Reject arrays whose row counts disagree, so every batch is internally consistent."""
        rows = self.rhs.shape[0]
        if self.rhs.ndim != 2 or self.solutions.shape != self.rhs.shape:
            raise ValueError(
                f"Batch rhs {self.rhs.shape} and solutions {self.solutions.shape} must be "
                "matching 2-D arrays."
            )
        if self.row_kind_codes.shape != (rows,) or self.matrix_sample_index.shape != (rows,):
            raise ValueError(
                f"Batch per-row arrays must have {rows} rows, got row_kind_codes "
                f"{self.row_kind_codes.shape} and matrix_sample_index "
                f"{self.matrix_sample_index.shape}."
            )

    def __len__(self) -> int:
        """Number of rows in the batch."""
        return int(self.rhs.shape[0])

    def slice(self, start: int, stop: int) -> SampleBatch:
        """Return the rows ``[start, stop)`` as a new batch with the same identity."""
        return SampleBatch(
            binding_index=self.binding_index,
            strategy_name=self.strategy_name,
            rhs=self.rhs[start:stop],
            solutions=self.solutions[start:stop],
            row_kind_codes=self.row_kind_codes[start:stop],
            matrix_sample_index=self.matrix_sample_index[start:stop],
            parameter_vectors=self.parameter_vectors,
        )


__all__ = ["SampleBatch"]
