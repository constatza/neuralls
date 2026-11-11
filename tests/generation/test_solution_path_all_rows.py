"""A ``.txt`` solution file holds one row, to the reader and to the archive row count alike."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.source_streams import open_vector_stream
from tests.fakes.archive_source import InMemoryArchiveSource

SOLUTION_PATH = Path("/archive/solutions.txt")
N_UNKNOWNS = 4
SEED = 7


@pytest.fixture
def archive() -> InMemoryArchiveSource:
    """In-memory archive holding one ``.txt`` solution vector."""
    source = InMemoryArchiveSource()
    rng = np.random.default_rng(SEED)
    source.add(SOLUTION_PATH, rng.standard_normal(N_UNKNOWNS))
    return source


def test_text_solution_file_row_semantics_match_reader(archive: InMemoryArchiveSource) -> None:
    """The row count of a ``.txt`` solution file equals the rows the vector reader loads."""
    reader_rows = len(
        open_vector_stream(
            str(SOLUTION_PATH), readers=archive.readers(), archive_source=archive
        ).sample_ids
    )

    assert archive.row_count(SOLUTION_PATH) == reader_rows
