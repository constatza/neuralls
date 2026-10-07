"""Write generated batches at a running row offset into pre-sized dense arrays.

The writer holds no sample data between calls. Each batch is written at the next free
row, so memory is bounded by the batch size rather than the dataset size. The array
names and the matrix/index encodings match the buffered accumulators, so readers see
the same layout either way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuralls.domain.generation.batch import SampleBatch
from neuralls.domain.generation.ports import ArrayStore
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX

RHS_NAME: str = "rhs"
SOLUTIONS_NAME: str = "solutions"
ROW_KIND_NAME: str = "row_kind"
MATRIX_SAMPLE_INDEX_NAME: str = "matrix_sample_index"
MATRIX_NAME: str = "matrix"

FLOAT_DTYPE: str = "float64"
ROW_KIND_DTYPE: str = "uint8"
MATRIX_SAMPLE_INDEX_DTYPE: str = "int64"

SINGLE_MATRIX_ROWS: int = 1
"""Physical matrix rows when one shared matrix is broadcast to every sample."""


@dataclass(frozen=True)
class StoredArray:
    """One array the writer created, with its final shape and dtype.

    Attributes:
        name: Array name in the store.
        shape: Full shape the array was created with.
        dtype: Element dtype name.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def rows(self) -> int:
        """Leading dimension, the sample count for every array the writer creates."""
        return self.shape[0]


@dataclass(frozen=True)
class DatasetArtifacts:
    """Description of every array the writer persisted.

    Attributes:
        arrays: Created arrays in creation order.
    """

    arrays: tuple[StoredArray, ...]

    def get(self, name: str) -> StoredArray | None:
        """Return the named array's description, or None when the writer did not create it."""
        return next((array for array in self.arrays if array.name == name), None)


MatrixLookup = Callable[[int], NDArray]
"""Maps a matrix sample id to its dense normalized matrix, shape (n, n)."""


class SampleWriter:
    """Append batches to dense arrays of known total length. Write-only by design.

    Arrays are created lazily at the first batch, because the feature dimension and the
    parameter widths are only known once data arrives. The row total comes from the plan,
    so every array is created at its final shape.

    TODO: the schema (array shapes, which parameter streams exist) is currently a side
    effect of the first ``write_batch`` call rather than a value fixed up front. A two-phase
    constructor — an ``open(first_batch, ...)`` that creates the arrays and returns a writer
    whose schema is then immutable — would make that explicit. Both streaming callers would
    need to find the first non-empty batch before constructing the writer, and the direct
    constructor-then-``write_batch`` tests in ``test_sample_writer.py`` would need rewriting.
    Worth doing if a second writer backend or a schema-inspection need appears; not before.
    """

    def __init__(
        self,
        store: ArrayStore,
        *,
        planned_rows: int,
        matrix_for: MatrixLookup | None,
        single_matrix: bool,
    ) -> None:
        """Bind the writer to a store and to the planned row total.

        Args:
            store: Backend that holds the arrays. The writer closes it in ``finalize``.
            planned_rows: Total rows the run will produce. Written rows must equal it.
            matrix_for: Dense matrix for a matrix sample id. None when the matrix is stored
                by another writer (CSR), in which case no ``matrix`` array is created.
            single_matrix: True when all rows share one matrix, stored as one broadcast row.
                Ignored when ``matrix_for`` is None.

        Raises:
            ValueError: If ``planned_rows`` is negative.
        """
        if planned_rows < 0:
            raise ValueError(f"planned_rows must be non-negative, got {planned_rows}")
        self._store = store
        self._planned_rows = planned_rows
        self._matrix_for = matrix_for
        self._single_matrix = single_matrix
        self._rows_written = 0
        self._created: list[StoredArray] = []
        self._parameter_streams: set[int] = set()
        self._single_matrix_written = False

    def write_batch(self, batch: SampleBatch) -> None:
        """Write one batch at the running offset.

        Args:
            batch: Rows of one strategy on one binding. Empty batches are skipped.

        Raises:
            ValueError: If the batch would exceed the planned row total, or carries a
                parameter stream that the first batch did not declare.
        """
        rows = len(batch)
        if rows == 0:
            return
        start = self._rows_written
        if start + rows > self._planned_rows:
            raise ValueError(
                f"Batch would write row {start + rows} but only {self._planned_rows} rows were planned."
            )
        if not self._created:
            self._create_arrays(batch)
        self._store.write_rows(RHS_NAME, start, _as_float(batch.rhs))
        self._store.write_rows(SOLUTIONS_NAME, start, _as_float(batch.solutions))
        self._store.write_rows(
            ROW_KIND_NAME, start, np.asarray(batch.row_kind_codes, dtype=np.uint8)
        )
        self._store.write_rows(
            MATRIX_SAMPLE_INDEX_NAME, start, np.asarray(batch.matrix_sample_index, dtype=np.int64)
        )
        self._write_matrix(batch, start, rows)
        self._write_parameters(batch, start, rows)
        self._rows_written += rows

    def finalize(self) -> DatasetArtifacts:
        """Check the written row count against the plan, close the store, and describe it.

        Raises:
            ValueError: If no rows were written, or the written rows differ from the plan.
                Incomplete arrays are reported by the store's ``close``.
        """
        if self._rows_written == 0:
            raise ValueError("No samples were generated for dataset persistence.")
        if self._rows_written != self._planned_rows:
            raise ValueError(
                f"Written rows {self._rows_written} differ from planned {self._planned_rows}."
            )
        self._store.close()
        return DatasetArtifacts(arrays=tuple(self._created))

    def _create_arrays(self, batch: SampleBatch) -> None:
        feature_dim = int(batch.rhs.shape[1])
        rows = self._planned_rows
        self._declare(RHS_NAME, (rows, feature_dim), FLOAT_DTYPE)
        self._declare(SOLUTIONS_NAME, (rows, feature_dim), FLOAT_DTYPE)
        self._declare(ROW_KIND_NAME, (rows,), ROW_KIND_DTYPE)
        self._declare(MATRIX_SAMPLE_INDEX_NAME, (rows,), MATRIX_SAMPLE_INDEX_DTYPE)
        if self._matrix_for is not None:
            matrix_rows = SINGLE_MATRIX_ROWS if self._single_matrix else rows
            self._declare(MATRIX_NAME, (matrix_rows, feature_dim, feature_dim), FLOAT_DTYPE)
        for index, vector in enumerate(batch.parameter_vectors):
            if vector is None:
                continue
            width = int(np.shape(vector)[-1])
            self._parameter_streams.add(index)
            self._declare(_parameter_name(index), (rows, width), FLOAT_DTYPE)

    def _declare(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        self._store.create(name, shape, dtype)
        self._created.append(StoredArray(name=name, shape=shape, dtype=dtype))

    def _write_matrix(self, batch: SampleBatch, start: int, rows: int) -> None:
        if self._matrix_for is None:
            return
        if self._single_matrix:
            if self._single_matrix_written:
                return
            self._single_matrix_written = True
            matrix = _as_float(self._matrix_for(int(batch.matrix_sample_index[0])))
            self._store.write_rows(MATRIX_NAME, 0, matrix[np.newaxis])
            return
        sample_id = int(batch.matrix_sample_index[0])
        matrix = _as_float(self._matrix_for(sample_id))
        self._store.write_rows(MATRIX_NAME, start, np.broadcast_to(matrix, (rows, *matrix.shape)))

    def _write_parameters(self, batch: SampleBatch, start: int, rows: int) -> None:
        for index, vector in enumerate(batch.parameter_vectors):
            if index not in self._parameter_streams:
                if vector is None:
                    continue
                raise ValueError(
                    f"Parameter stream {index} appears after the first batch, "
                    "so its array was never created."
                )
            if vector is None:
                raise ValueError(
                    f"Parameter stream {index} has no sample in a batch after the first, "
                    "which would leave its rows unwritten."
                )
            tiled = np.tile(np.asarray(vector, dtype=np.float64), (rows, 1))
            self._store.write_rows(_parameter_name(index), start, tiled)


def _parameter_name(index: int) -> str:
    return f"{PARAMETERS_ZARR_PREFIX}{index}"


def _as_float(array: NDArray) -> NDArray[np.float64]:
    return np.asarray(array, dtype=np.float64)


__all__ = [
    "DatasetArtifacts",
    "MatrixLookup",
    "SampleWriter",
    "StoredArray",
]
