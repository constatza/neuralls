"""Incremental CSR writer: running offsets over a write-only member sink.

The writer keeps the same member schema as the buffered writers in ``layout.py``.
It knows the planned sample count and layout up front, so every per-sample member
is created at its final shape. The flat ``indptr``, ``indices`` and ``data`` of the
ragged layout are growable and are appended batch by batch. The writer never reads
from the sink; the sink only creates, fills and extends members.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol

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


class _LayoutStrategy(Protocol):
    """Writes one batch of samples at a running offset, creating members on first write.

    Each strategy owns the invariants its layout needs to track between batches (the
    created-members guard, plus the ragged nnz cursor or the shared pattern) — nothing
    the writer itself needs to branch on.
    """

    def write(
        self, sink: CsrMemberSink, samples: Sequence[csr_array], start: int, planned: int
    ) -> None: ...


class _RaggedLayout:
    """``MANY_MATRICES``: every sample keeps its own pattern, appended to flat CSR arrays."""

    def __init__(self) -> None:
        self._created = False
        self._nnz_end = _NO_NNZ

    def write(
        self, sink: CsrMemberSink, samples: Sequence[csr_array], start: int, planned: int
    ) -> None:
        batch = pack_per_sample(samples)
        if not self._created:
            self._create(sink, planned)
        sink.append(INDPTR_ARRAY, batch.indptr)
        sink.append(INDICES_ARRAY, batch.indices)
        sink.append(DATA_ARRAY, batch.data)
        sink.write_rows(SAMPLE_OFFSETS_ARRAY, start + 1, batch.sample_offsets[1:] + self._nnz_end)
        sink.write_rows(SHAPE_ARRAY, start, batch.shape)
        self._nnz_end += int(batch.data.shape[0])

    def _create(self, sink: CsrMemberSink, planned: int) -> None:
        sink.create_growable(INDPTR_ARRAY, np.dtype(INDEX_DTYPE))
        sink.create_growable(INDICES_ARRAY, np.dtype(INDEX_DTYPE))
        sink.create_growable(DATA_ARRAY, np.dtype(VALUE_DTYPE))
        sink.create_fixed(SAMPLE_OFFSETS_ARRAY, (planned + 1,), np.dtype(INDEX_DTYPE))
        sink.create_fixed(SHAPE_ARRAY, (planned, SHAPE_COLUMNS), np.dtype(INDEX_DTYPE))
        sink.write_rows(SAMPLE_OFFSETS_ARRAY, _FIRST_SAMPLE, np.zeros(1, dtype=INDEX_DTYPE))
        self._created = True


class _SharedLayout:
    """``SHARED_PATTERN``: one pattern shared by every sample, rejecting any that differs."""

    def __init__(self) -> None:
        self._created = False
        self._pattern: csr_array | None = None

    def write(
        self, sink: CsrMemberSink, samples: Sequence[csr_array], start: int, planned: int
    ) -> None:
        canonical = [canonicalise_csr(sample) for sample in samples]
        if self._pattern is None:
            self._pattern = canonical[_FIRST_SAMPLE]
        for offset, sample in enumerate(canonical):
            if not csr_pattern_matches(self._pattern, sample):
                raise ValueError(
                    f"shared-pattern stream: stored sample {start + offset} has a "
                    f"sparsity pattern that differs from sample {_FIRST_SAMPLE}; set "
                    '[output].sparsity_pattern = "ragged" when matrices differ in pattern'
                )
        parts = [to_components(sample) for sample in canonical]
        if not self._created:
            self._create(sink, planned, to_components(self._pattern))
        data = np.stack([part.data for part in parts]).astype(VALUE_DTYPE, copy=False)
        shape = shape_table([part.shape for part in parts])
        sink.write_rows(DATA_ARRAY, start, data)
        sink.write_rows(SHAPE_ARRAY, start, shape)

    def _create(self, sink: CsrMemberSink, planned: int, pattern: CsrComponents) -> None:
        nnz = int(pattern.data.shape[0])
        sink.create_fixed(INDPTR_ARRAY, pattern.indptr.shape, np.dtype(INDEX_DTYPE))
        sink.write_rows(INDPTR_ARRAY, _FIRST_SAMPLE, pattern.indptr)
        sink.create_fixed(INDICES_ARRAY, (nnz,), np.dtype(INDEX_DTYPE))
        sink.write_rows(INDICES_ARRAY, _FIRST_SAMPLE, pattern.indices)
        sink.create_fixed(DATA_ARRAY, (planned, nnz), np.dtype(VALUE_DTYPE))
        sink.create_fixed(SHAPE_ARRAY, (planned, SHAPE_COLUMNS), np.dtype(INDEX_DTYPE))
        self._created = True


_LAYOUT_STRATEGIES: Final[Mapping[LayoutType, Callable[[], _LayoutStrategy]]] = {
    _RAGGED_LAYOUT: _RaggedLayout,
    _SHARED_LAYOUT: _SharedLayout,
}


class CsrStreamWriter:
    """Streams CSR samples into one container through a ``CsrMemberSink``.

    The layout is chosen once, as a ``_LayoutStrategy``, and never branched on again; members
    are created lazily by that strategy on its first batch, because the shared-pattern member
    shapes depend on the pattern, which is only known once samples arrive.
    """

    def __init__(self, sink: CsrMemberSink, planned_samples: int, layout: LayoutType) -> None:
        if planned_samples < 1:
            raise ValueError(
                f"a streamed CSR dataset needs at least one sample, got {planned_samples}"
            )
        try:
            strategy_for_layout = _LAYOUT_STRATEGIES[layout]
        except KeyError:
            raise ValueError(
                f"streamed CSR supports MANY_MATRICES and SHARED_PATTERN, got {layout}"
            ) from None
        self._sink = sink
        self._planned = planned_samples
        self._layout = layout
        self._written = 0
        self._strategy = strategy_for_layout()

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
        self._strategy.write(self._sink, samples, self._written, self._planned)
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
