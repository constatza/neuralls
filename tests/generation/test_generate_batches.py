"""Batched generation must emit the pre-shuffle row order of the accumulation path.

The golden digests were captured from the pre-batching pipeline (``shuffle=False``,
three SPD matrices, gaussian_forward + gaussian_residuals). Batch size only changes how
the rows are chunked, never their order or content, so every size must match them.
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
from neuralls.shared.types import MatrixFormat

_FLOAT_DIGEST_DECIMALS = 12
_NUM_MATRICES = 3
_GOLDEN_DIGESTS: dict[str, str] = {
    "rhs": "1ed7b231314cb7ca",
    "solutions": "2cf722daad931872",
    "row_kind_codes": "62a7c5d9837d645b",
    "matrix_sample_index": "c3af06584c6a3d1a",
}


def _digest(array: np.ndarray) -> str:
    """Stable content digest of an array's dtype, shape and (rounded) bytes."""
    contiguous = np.ascontiguousarray(array)
    if np.issubdtype(contiguous.dtype, np.floating):
        contiguous = np.round(contiguous, decimals=_FLOAT_DIGEST_DECIMALS)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode())
    hasher.update(str(contiguous.shape).encode())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()[:16]


@pytest.fixture
def spd_matrix_dir(tmp_path: Path) -> Path:
    """Three deterministic 5x5 SPD matrices as individually globbable .txt files."""
    rng = np.random.default_rng(11)
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    for index in range(_NUM_MATRICES):
        base = rng.standard_normal((5, 5))
        np.savetxt(matrix_dir / f"A_{index:03d}.txt", base @ base.T + 5.0 * np.eye(5))
    return matrix_dir


@pytest.fixture
def generation_order_source(spd_matrix_dir: Path) -> SourceSpec:
    return SourceSpec(matrix_path=str(spd_matrix_dir / "A_*.txt"))


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
        normalize="matrix",
    )


def _stream_batches(
    source: SourceSpec, spec: DatasetSpec, batch_size: int
) -> Iterator[SampleBatch]:
    """Run the batch pipeline for one source and spec, as the orchestrator wires it."""
    context = _prepare_generation_context(source, spec)
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
    batch_size: int,
) -> None:
    batches = list(_stream_batches(generation_order_source, generation_order_spec, batch_size))
    arrays = _concatenate(batches)

    digests = {name: _digest(array) for name, array in arrays.items()}
    assert digests == _GOLDEN_DIGESTS


@pytest.mark.parametrize("batch_size", [1, 4, 1024])
def test_generate_batches_never_exceeds_batch_size(
    generation_order_source: SourceSpec,
    generation_order_spec: DatasetSpec,
    batch_size: int,
) -> None:
    batches = list(_stream_batches(generation_order_source, generation_order_spec, batch_size))

    assert all(len(batch.rhs) <= batch_size for batch in batches)
    assert sum(len(batch.rhs) for batch in batches) == len(_concatenate(batches)["rhs"])


def test_generate_batches_rejects_non_positive_batch_size(
    generation_order_source: SourceSpec,
    generation_order_spec: DatasetSpec,
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        list(_stream_batches(generation_order_source, generation_order_spec, batch_size=0))
