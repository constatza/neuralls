"""Batched generation must emit the pre-shuffle row order of the accumulation path.

The golden values were captured from the pre-batching pipeline (``shuffle=False``,
three SPD matrices, gaussian_forward + gaussian_residuals): exact digests of the integer
row kinds and matrix indices, and the norm of every float row, compared within a
tolerance that BLAS rounding differences stay under. Batch size only changes how the rows
are chunked, never their order or content, so every size must match them and reproduce
the rows of one-row batches.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuralls.domain.generation.batch import SampleBatch
from neuralls.domain.generation.batch_generator import generate_batches
from neuralls.domain.generation.batch_plan import plan_batches
from neuralls.domain.generation.matrix_cache import _cached_matrix_loader
from neuralls.domain.generation.orchestration import (
    _make_strategy_runner,
    _prepare_generation_context,
)
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.shared.system_scaling import GershgorinScaling
from neuralls.shared.types import MatrixFormat
from tests.fakes.archive_source import InMemoryArchiveSource

MATRIX_DIR = Path("archive/matrices")

_FLOAT_RTOL = 1e-10
"""Relative tolerance on float rows: the CG residual rows carry a few rounding steps, so
this sits well above the ~1e-14 drift of differing BLAS builds and far below any change."""
_GOLDEN_ROW_NORMS: dict[str, tuple[float, ...]] = {
    "rhs": (
        0.26338671314991224, 0.3969597240568228, 0.4869905272227404, 0.3394472961872561,
        0.07322622763918255, 0.02036401756337476, 0.00296989183486501, 0.29635072786682887,
        0.3752047093962407, 0.0394373524353824, 0.012955205376413693, 0.0026131475367018077,
    ),
    "solutions": (
        1.1403862977121118, 2.277021248314017, 1.7639366376356738, 1.5460354792771318,
        0.28347002059620446, 0.10186534533375381, 0.00877286485380757, 1.119466401551306,
        1.6829061345451362, 0.19823432732234725, 0.0458018517302848, 0.011385848696708023,
    ),
}  # fmt: skip
_NUM_MATRICES = 3
_GOLDEN_INTEGER_DIGESTS: dict[str, str] = {
    "row_kind_codes": "62a7c5d9837d645b",
    "matrix_sample_index": "c3af06584c6a3d1a",
}
_FLOAT_ARRAYS = tuple(_GOLDEN_ROW_NORMS)


def _digest(array: np.ndarray) -> str:
    """Content digest of an integer array's dtype, shape and bytes."""
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode())
    hasher.update(str(contiguous.shape).encode())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()[:16]


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """Three deterministic 5x5 SPD matrices as individually globbable in-memory files."""
    source = InMemoryArchiveSource()
    rng = np.random.default_rng(11)
    for index in range(_NUM_MATRICES):
        base = rng.standard_normal((5, 5))
        source.add(MATRIX_DIR / f"A_{index:03d}.txt", base @ base.T + 5.0 * np.eye(5))
    return source


@pytest.fixture
def generation_order_source() -> SourceSpec:
    return SourceSpec(matrix_path=str(MATRIX_DIR / "A_*.txt"))


@pytest.fixture
def generation_order_spec(solver_overrides: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        mixture=MixtureSpec(
            counts={"gaussian_forward": 6, "gaussian_residuals": 6},
            seed=1234,
            shuffle=False,
            strategy_overrides={"gaussian_residuals": {"stop": 3, "start": 1}},
            solver_overrides=solver_overrides,
        ),
        scaling=GershgorinScaling(),
    )


def _stream_batches(
    source: SourceSpec, spec: DatasetSpec, archive_source: InMemoryArchiveSource, batch_size: int
) -> Iterator[SampleBatch]:
    """Run the batch pipeline for one source and spec, as the orchestrator wires it."""
    context = _prepare_generation_context(
        source, spec, readers=archive_source.readers(), archive_source=archive_source
    )
    get_matrix = _cached_matrix_loader(context.streams.matrix, spec, MatrixFormat.DENSE)
    plan = plan_batches(context.allocation)
    runner = _make_strategy_runner(context, get_matrix, spec.mixture, lambda _index, _cached: None)
    return generate_batches(plan, runner, batch_size=batch_size)


def _concatenate(batches: Sequence[SampleBatch]) -> dict[str, np.ndarray]:
    return {
        "rhs": np.vstack([b.rhs for b in batches]),
        "solutions": np.vstack([b.solutions for b in batches]),
        "row_kind_codes": np.concatenate([b.row_kind_codes for b in batches]),
        "matrix_sample_index": np.concatenate([b.matrix_sample_index for b in batches]),
    }


@pytest.mark.parametrize("batch_size", [1, 4, 1024])
def test_generate_batches_order_matches_current_output(
    generation_order_source: SourceSpec,
    generation_order_spec: DatasetSpec,
    archive_source: InMemoryArchiveSource,
    batch_size: int,
) -> None:
    def generate(size: int) -> dict[str, np.ndarray]:
        return _concatenate(
            list(
                _stream_batches(
                    generation_order_source, generation_order_spec, archive_source, size
                )
            )
        )

    arrays = generate(batch_size)
    one_row_batches = generate(1)

    digests = {name: _digest(arrays[name]) for name in _GOLDEN_INTEGER_DIGESTS}
    assert digests == _GOLDEN_INTEGER_DIGESTS
    for name in _FLOAT_ARRAYS:
        np.testing.assert_allclose(
            np.linalg.norm(arrays[name], axis=1), _GOLDEN_ROW_NORMS[name], rtol=_FLOAT_RTOL
        )
        np.testing.assert_allclose(
            arrays[name],
            one_row_batches[name],
            rtol=_FLOAT_RTOL,
            atol=_FLOAT_RTOL * np.abs(one_row_batches[name]).max(),
        )


@pytest.mark.parametrize("batch_size", [1, 4, 1024])
def test_generate_batches_never_exceeds_batch_size(
    generation_order_source: SourceSpec,
    generation_order_spec: DatasetSpec,
    archive_source: InMemoryArchiveSource,
    batch_size: int,
) -> None:
    batches = list(
        _stream_batches(generation_order_source, generation_order_spec, archive_source, batch_size)
    )

    assert all(len(batch.rhs) <= batch_size for batch in batches)
    assert sum(len(batch.rhs) for batch in batches) == len(_concatenate(batches)["rhs"])


def test_generate_batches_rejects_non_positive_batch_size(
    generation_order_source: SourceSpec,
    generation_order_spec: DatasetSpec,
    archive_source: InMemoryArchiveSource,
) -> None:
    with pytest.raises(ValueError):
        list(
            _stream_batches(
                generation_order_source, generation_order_spec, archive_source, batch_size=0
            )
        )
