"""Per-binding inputs: parameter vectors are float64 and each matrix is loaded once."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.orchestration import BatchStream, open_batch_stream
from neuralls.domain.generation.source_streams import _MatrixStream
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.shared.types import MatrixFormat, SystemMatrix
from tests.fakes.archive_source import InMemoryArchiveSource

SEED = 7
N_UNKNOWNS = 4
NUM_MATRICES = 3
PARAMETER_LENGTH = 4
SOURCE_REGEX = r"(\d+)(?!.*\d)"
MATRIX_DIR = Path("/archive/matrices")
PARAMETER_DIR = Path("/archive/parameters")


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """Three seeded SPD matrices ``A_0.txt``..``A_2.txt`` and two integer parameter files."""
    source = InMemoryArchiveSource()
    rng = np.random.default_rng(SEED)
    for i in range(NUM_MATRICES):
        base = rng.standard_normal((N_UNKNOWNS, N_UNKNOWNS))
        source.add(MATRIX_DIR / f"A_{i}.txt", base @ base.T + N_UNKNOWNS * np.eye(N_UNKNOWNS))
    for i in range(2):
        source.add(PARAMETER_DIR / f"p_{i}.npy", np.arange(PARAMETER_LENGTH, dtype=np.int64) + i)
    return source


@pytest.fixture
def open_binding_stream(
    archive_source: InMemoryArchiveSource,
) -> Callable[[SourceSpec, dict[str, int]], BatchStream]:
    """Open a batch stream over the in-memory archive from a source and per-strategy counts."""

    def _open(source: SourceSpec, counts: dict[str, int]) -> BatchStream:
        spec = DatasetSpec(mixture=MixtureSpec(counts=counts, seed=SEED, shuffle=False))
        return open_batch_stream(
            source,
            spec,
            batch_size=1 << 20,
            matrix_format=MatrixFormat.DENSE,
            readers=archive_source.readers(),
            archive_source=archive_source,
        )

    return _open


def test_parameter_vectors_are_float64_for_integer_files(
    open_binding_stream: Callable[[SourceSpec, dict[str, int]], BatchStream],
) -> None:
    """An int-valued parameter file yields float64 parameter vectors, as RHS and solutions do."""
    source = SourceSpec(
        matrix_path=str(MATRIX_DIR / "A_[01].txt"),
        sample_id_regex=SOURCE_REGEX,
        parameters_paths=(str(PARAMETER_DIR / "p_*.npy"),),
    )
    stream = open_binding_stream(source, {"neutral_ones": 2})

    batches = list(stream.batches)

    assert batches
    for batch in batches:
        vector = batch.parameter_vectors[0]
        assert vector is not None
        assert vector.dtype == np.float64


def test_each_matrix_is_loaded_once_across_strategies_and_bindings(
    monkeypatch: pytest.MonkeyPatch,
    open_binding_stream: Callable[[SourceSpec, dict[str, int]], BatchStream],
) -> None:
    """Two strategies over three bindings load each matrix sample exactly once."""
    loads: Counter[int] = Counter()
    real_load: Callable[[_MatrixStream, int, MatrixFormat], SystemMatrix] = (
        _MatrixStream.load_sample
    )

    def _counting_load(
        self: _MatrixStream, sample_id: int, matrix_format: MatrixFormat
    ) -> SystemMatrix:
        loads[sample_id] += 1
        return real_load(self, sample_id, matrix_format)

    monkeypatch.setattr(_MatrixStream, "load_sample", _counting_load)
    source = SourceSpec(
        matrix_path=str(MATRIX_DIR / "A_*.txt"),
        sample_id_regex=SOURCE_REGEX,
    )
    stream = open_binding_stream(source, {"neutral_ones": 3, "gaussian_forward": 3})

    list(stream.batches)

    assert loads == Counter({0: 1, 1: 1, 2: 1})
