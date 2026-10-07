"""Incremental CSR writer: running offsets over a write-only member sink.

The writer keeps the same member schema as the buffered writers in ``layout.py``.
It knows the planned sample count and layout up front, so every per-sample member
is created at its final shape. The flat ``indptr``, ``indices`` and ``data`` of the
ragged layout are growable and are appended batch by batch. The writer never reads
from the sink; the sink only creates, fills and extends members.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.components import (
    INDEX_DTYPE,
    VALUE_DTYPE,
    CsrComponents,
    to_components,
)
from neuralls.platform.sparse_io.layout import (
    DATA_ARRAY,
    INDICES_ARRAY,
    INDPTR_ARRAY,
    SAMPLE_OFFSETS_ARRAY,
    SHAPE_ARRAY,
    SHAPE_COLUMNS,
    pack_per_sample,
    shape_table,
)
from neuralls.platform.sparse_io.pattern import canonicalise_csr, csr_pattern_matches
from neuralls.platform.sparse_io.protocol import CsrMemberSink, SparseSummary
from neuralls.shared.types import LayoutType

_SHARED_LAYOUT: Final = LayoutType.SHARED_PATTERN
_RAGGED_LAYOUT: Final = LayoutType.MANY_MATRICES
_FIRST_SAMPLE: Final = 0
_NO_NNZ: Final = 0


class CsrStreamWriter:
    """Streams CSR samples into one container through a ``CsrMemberSink``.

    Members are created lazily on the first batch, because the shared-pattern
    member shapes depend on the pattern, which is only known once samples arrive.
    """

    def __init__(self, sink: CsrMemberSink, planned_samples: int, layout: LayoutType) -> None:
        if planned_samples < 1:
            raise ValueError(
                f"a streamed CSR dataset needs at least one sample, got {planned_samples}"
            )
        if layout not in (_RAGGED_LAYOUT, _SHARED_LAYOUT):
            raise ValueError(
                f"streamed CSR supports MANY_MATRICES and SHARED_PATTERN, got {layout}"
            )
        self._sink = sink
        self._planned = planned_samples
        self._layout = layout
        self._written = 0
        self._nnz_end = _NO_NNZ
        self._pattern: csr_array | None = None
        self._created = False

    def write_batch(self, samples: Sequence[csr_array]) -> None:
        """Append one batch of samples at the running offsets.

        Raises:
            ValueError: If the batch is empty, would exceed the planned sample count, or
                (shared layout) does not match the pattern of the first sample.
        """
        if not samples:
            raise ValueError("cannot write an empty sample batch")
        if self._written + len(samples) > self._planned:
            raise ValueError(
                f"batch of {len(samples)} samples exceeds the planned total of {self._planned} "
                f"with {self._written} already written"
            )
        if self._layout is _SHARED_LAYOUT:
            self._write_shared(samples)
        else:
            self._write_ragged(samples)
        self._written += len(samples)

    def close(self) -> SparseSummary:
        """Check the planned sample count and release the sink.

        Raises:
            ValueError: If fewer or more samples were written than planned.
        """
        try:
            if self._written != self._planned:
                raise ValueError(
                    f"planned {self._planned} samples but wrote {self._written}; the source "
                    "produced fewer samples than its plan"
                )
        finally:
            self._sink.close()
        return SparseSummary(sample_count=self._written, layout=self._layout)

    def _write_ragged(self, samples: Sequence[csr_array]) -> None:
        batch = pack_per_sample(samples)
        if not self._created:
            self._create_ragged()
        start = self._written
        self._sink.append(INDPTR_ARRAY, batch.indptr)
        self._sink.append(INDICES_ARRAY, batch.indices)
        self._sink.append(DATA_ARRAY, batch.data)
        self._sink.write_rows(
            SAMPLE_OFFSETS_ARRAY, start + 1, batch.sample_offsets[1:] + self._nnz_end
        )
        self._sink.write_rows(SHAPE_ARRAY, start, batch.shape)
        self._nnz_end += int(batch.data.shape[0])

    def _write_shared(self, samples: Sequence[csr_array]) -> None:
        canonical = [canonicalise_csr(sample) for sample in samples]
        if self._pattern is None:
            self._pattern = canonical[_FIRST_SAMPLE]
        for offset, sample in enumerate(canonical):
            if not csr_pattern_matches(self._pattern, sample):
                raise ValueError(
                    f"shared-pattern stream: stored sample {self._written + offset} has a "
                    f"sparsity pattern that differs from sample {_FIRST_SAMPLE}; set "
                    '[output].sparsity_pattern = "ragged" when matrices differ in pattern'
                )
        parts = [to_components(sample) for sample in canonical]
        if not self._created:
            self._create_shared(to_components(self._pattern))
        start = self._written
        data = np.stack([part.data for part in parts]).astype(VALUE_DTYPE, copy=False)
        shape = shape_table([part.shape for part in parts])
        self._sink.write_rows(DATA_ARRAY, start, data)
        self._sink.write_rows(SHAPE_ARRAY, start, shape)

    def _create_ragged(self) -> None:
        self._sink.create_growable(INDPTR_ARRAY, np.dtype(INDEX_DTYPE))
        self._sink.create_growable(INDICES_ARRAY, np.dtype(INDEX_DTYPE))
        self._sink.create_growable(DATA_ARRAY, np.dtype(VALUE_DTYPE))
        self._sink.create_fixed(SAMPLE_OFFSETS_ARRAY, (self._planned + 1,), np.dtype(INDEX_DTYPE))
        self._sink.create_fixed(SHAPE_ARRAY, (self._planned, SHAPE_COLUMNS), np.dtype(INDEX_DTYPE))
        self._sink.write_rows(SAMPLE_OFFSETS_ARRAY, _FIRST_SAMPLE, np.zeros(1, dtype=INDEX_DTYPE))
        self._created = True

    def _create_shared(self, pattern: CsrComponents) -> None:
        nnz = int(pattern.data.shape[0])
        self._sink.create_fixed(INDPTR_ARRAY, pattern.indptr.shape, np.dtype(INDEX_DTYPE))
        self._sink.write_rows(INDPTR_ARRAY, _FIRST_SAMPLE, pattern.indptr)
        self._sink.create_fixed(INDICES_ARRAY, (nnz,), np.dtype(INDEX_DTYPE))
        self._sink.write_rows(INDICES_ARRAY, _FIRST_SAMPLE, pattern.indices)
        self._sink.create_fixed(DATA_ARRAY, (self._planned, nnz), np.dtype(VALUE_DTYPE))
        self._sink.create_fixed(SHAPE_ARRAY, (self._planned, SHAPE_COLUMNS), np.dtype(INDEX_DTYPE))
        self._created = True
