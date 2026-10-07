"""Per-binding inputs: parameter vectors are float64 and each matrix is loaded once."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.orchestration import BatchStream
from neuralls.domain.generation.source_streams import _MatrixStream
from neuralls.domain.generation.specs import SourceSpec
from neuralls.shared.types import MatrixFormat, SystemMatrix

PARAMETER_LENGTH = 4
NEUTRAL_COUNT = 1


@pytest.fixture
def integer_parameter_dir(tmp_path: Path) -> Path:
    """One integer-valued parameter file per matrix, named to pair by sample id."""
    param_dir = tmp_path / "parameters"
    param_dir.mkdir()
    for i in range(2):
        np.save(param_dir / f"p_{i}.npy", np.arange(PARAMETER_LENGTH, dtype=np.int64) + i)
    return param_dir


def test_parameter_vectors_are_float64_for_integer_files(
    two_matrix_source: SourceSpec,
    integer_parameter_dir: Path,
    open_binding_stream: Callable[..., BatchStream],
) -> None:
    """An int-valued parameter file yields float64 parameter vectors, as RHS and solutions do."""
    source = SourceSpec(
        matrix_path=two_matrix_source.matrix_path,
        sample_id_regex=two_matrix_source.sample_id_regex,
        parameters_paths=(str(integer_parameter_dir / "p_*.npy"),),
    )
    stream = open_binding_stream(source, {"neutral_ones": 2})

    batches = list(stream.batches)

    assert batches
    for batch in batches:
        vector = batch.parameter_vectors[0]
        assert vector is not None
        assert vector.dtype == np.float64


def test_each_matrix_is_loaded_once_across_strategies_and_bindings(
    three_spd_matrix_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_binding_stream: Callable[..., BatchStream],
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
        matrix_path=str(three_spd_matrix_dir / "A_*.txt"),
        sample_id_regex=r"(\d+)(?!.*\d)",
    )
    stream = open_binding_stream(source, {"neutral_ones": 3, "gaussian_forward": 3})

    list(stream.batches)

    assert loads == Counter({0: 1, 1: 1, 2: 1})
