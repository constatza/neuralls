"""Generation outcomes reach the operator log once each, with their facts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neuralls.composition.generation.generation_log import (
    log_planned_notices,
    logged_batches,
    render_row_cap,
    render_solve_notice,
)
from neuralls.domain.generation.batch import SampleBatch
from neuralls.domain.generation.batch_plan import BatchPlan
from neuralls.domain.generation.notices import (
    InvalidPairs,
    RowCapSource,
    RowsCapped,
    SolveNotConverged,
    SolveNotice,
)
from neuralls.domain.generation.orchestration import BatchStream
from neuralls.domain.generation.scalar_aggregate import ScalarAggregator

type CapturedLog = Callable[[str], list[str]]

_CAP = RowsCapped("solution_archive", 20, 15, RowCapSource.ARCHIVE_PAIRS)
_NOT_CONVERGED = SolveNotConverged(row=1, exit_code=500, residual=1.5e-3)
_CAP_LINE = (
    "Strategy 'solution_archive' requested 20 rows but its archive (matrix, file) pairs "
    "supply 15 rows: emitting 15 rows."
)
_NOT_CONVERGED_LINE = (
    "Strategy 'gaussian_inverse' (binding 0): CG did not converge on row 1 "
    "(exit_code=500, residual: 1.50e-03)."
)


@pytest.fixture
def noticed_batch() -> SampleBatch:
    """Four rows of one strategy on binding 0, carrying one solve notice."""
    return SampleBatch(
        binding_index=0,
        strategy_name="gaussian_inverse",
        rhs=np.zeros((4, 3)),
        solutions=np.zeros((4, 3)),
        row_kind_codes=np.zeros(4, dtype=np.uint8),
        matrix_sample_index=np.zeros(4, dtype=np.int64),
        parameter_vectors=(),
        notices=(_NOT_CONVERGED,),
    )


@pytest.fixture
def capped_stream(noticed_batch: SampleBatch) -> BatchStream:
    """A planned cap and the noticed batch cut into two slices, as generate_batches cuts it."""
    return BatchStream(
        plan=BatchPlan(bindings=()),
        single_matrix=True,
        batches=iter((noticed_batch.slice(0, 2), noticed_batch.slice(2, 4))),
        matrix_for=lambda sample_id: np.eye(3),
        scale=ScalarAggregator(),
        notices=(_CAP,),
    )


@pytest.mark.parametrize(
    ("notice", "expected"),
    [
        (_CAP, _CAP_LINE),
        (
            RowsCapped("solution_archive", 6, 4, RowCapSource.SOLUTION_FILE),
            (
                "Strategy 'solution_archive' requested 6 rows per binding but its solution file "
                "has 4 rows: emitting 4 rows per binding."
            ),
        ),
    ],
)
def test_row_cap_states_its_unit(notice: RowsCapped, expected: str) -> None:
    assert render_row_cap(notice) == expected


@pytest.mark.parametrize(
    ("notice", "expected"),
    [
        (_NOT_CONVERGED, _NOT_CONVERGED_LINE),
        (
            InvalidPairs(count=3, total=10, max_error=0.25, tolerance=1e-10),
            (
                "Strategy 'gaussian_inverse' (binding 0): found 3/10 invalid pairs "
                "(max error: 2.50e-01 > tolerance: 1.00e-10); kept."
            ),
        ),
    ],
)
def test_every_solve_notice_renders_one_line(notice: SolveNotice, expected: str) -> None:
    assert render_solve_notice("gaussian_inverse", 0, notice) == expected


def test_only_the_first_slice_keeps_the_notices(noticed_batch: SampleBatch) -> None:
    assert [noticed_batch.slice(start, start + 2).notices for start in (0, 2)] == [
        (_NOT_CONVERGED,),
        (),
    ]


def test_each_notice_is_logged_once(capped_stream: BatchStream, captured_log: CapturedLog) -> None:
    warnings = captured_log("WARNING")

    log_planned_notices(capped_stream)
    written = list(logged_batches(capped_stream))

    assert [len(batch) for batch in written] == [2, 2]
    assert warnings == [_CAP_LINE, _NOT_CONVERGED_LINE]
