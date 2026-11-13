"""Generation notices render to one operator log line each."""

from __future__ import annotations

import pytest

from neuralls.composition.generation.generation_log import render_notice
from neuralls.domain.generation.notices import (
    GenerationNotice,
    InvalidPairs,
    RowCapSource,
    RowsCapped,
    SolveNotConverged,
)


@pytest.mark.parametrize(
    ("binding_index", "notice", "expected"),
    [
        (
            None,
            RowsCapped("solution_archive", 20, 15, RowCapSource.ARCHIVE_PAIRS),
            (
                "Strategy 'solution_archive' requested 20 rows but its archive (matrix, file) "
                "pairs hold 15: emitting 15 rows."
            ),
        ),
        (
            2,
            SolveNotConverged(row=4, exit_code=500, residual=1.5e-3),
            (
                "Strategy 'solution_archive' (binding 2): CG did not converge on row 4 "
                "(exit_code=500, residual: 1.50e-03)."
            ),
        ),
        (
            0,
            InvalidPairs(count=3, total=10, max_error=0.25, tolerance=1e-10),
            (
                "Strategy 'solution_archive' (binding 0): found 3/10 invalid pairs "
                "(max error: 2.50e-01 > tolerance: 1.00e-10); kept."
            ),
        ),
    ],
)
def test_every_notice_renders_one_line(
    binding_index: int | None, notice: GenerationNotice, expected: str
) -> None:
    assert render_notice("solution_archive", binding_index, notice) == expected
