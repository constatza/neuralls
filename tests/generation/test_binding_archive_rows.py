"""Explicit solution files feed archive strategies only, and only by their declared rule.

Cyclic counts draw rows ``(b + p) mod K`` for binding ``b``; ``samples=-1`` takes every row.
A generated strategy never receives archive rows, whatever its count.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.batch_plan import ALL_SAMPLES
from neuralls.domain.generation.orchestration import BatchStream, open_batch_stream
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.shared.types import MatrixFormat
from tests import generation_support
from tests.fakes.archive_source import InMemoryArchiveSource

CYCLIC_COUNT = 2
OVER_CAP_COUNT = 6
MATRIX_DIR = Path("/archive/matrices")
SOLUTION_PATH = Path("/archive/solutions.npy")


@pytest.fixture
def archive_source(solution_block: np.ndarray) -> InMemoryArchiveSource:
    """The shared binding matrices ``A_0.txt``..``A_2.txt`` and one stacked solution file."""
    source = InMemoryArchiveSource()
    for i, matrix in enumerate(generation_support.binding_matrices()):
        source.add(MATRIX_DIR / f"A_{i}.txt", matrix)
    source.add(SOLUTION_PATH, solution_block)
    return source


@pytest.fixture
def explicit_solution_source() -> SourceSpec:
    """Three matrices, each binding paired with the one explicit solution file."""
    return SourceSpec(
        matrix_path=str(MATRIX_DIR / "A_*.txt"),
        solution_path=str(SOLUTION_PATH),
        sample_id_regex=generation_support.BINDING_SOURCE_REGEX,
    )


@pytest.fixture
def open_binding_stream(
    archive_source: InMemoryArchiveSource,
) -> Callable[[SourceSpec, dict[str, int]], BatchStream]:
    """Open a batch stream over the in-memory archive from a source and per-strategy counts."""

    def _open(source: SourceSpec, counts: dict[str, int]) -> BatchStream:
        spec = DatasetSpec(
            mixture=MixtureSpec(counts=counts, seed=generation_support.BINDING_SEED, shuffle=False)
        )
        return open_batch_stream(
            source,
            spec,
            batch_size=1 << 20,
            matrix_format=MatrixFormat.DENSE,
            readers=archive_source.readers(),
            archive_source=archive_source,
        )

    return _open


def _binding_solutions(stream: BatchStream, n_bindings: int) -> list[np.ndarray]:
    """Solution rows of each binding, concatenated over that binding's batches in order."""
    batches = list(stream.batches)
    return [
        np.concatenate([b.solutions for b in batches if b.binding_index == index])
        for index in range(n_bindings)
    ]


def test_cyclic_count_draws_rows_mod_k_per_binding(
    explicit_solution_source: SourceSpec,
    open_binding_stream: Callable[[SourceSpec, dict[str, int]], BatchStream],
    solution_block: np.ndarray,
) -> None:
    """Explicit ``solution_archive`` count N over 3 matrices: binding b takes rows (b+p) mod K."""
    stream = open_binding_stream(explicit_solution_source, {"solution_archive": CYCLIC_COUNT})
    k = solution_block.shape[0]

    per_binding = _binding_solutions(stream, n_bindings=3)

    for binding_index, solutions in enumerate(per_binding):
        expected = solution_block[[(binding_index + p) % k for p in range(CYCLIC_COUNT)]]
        np.testing.assert_allclose(solutions, expected)


def test_cyclic_count_above_file_rows_warns_once(
    explicit_solution_source: SourceSpec,
    open_binding_stream: Callable[[SourceSpec, dict[str, int]], BatchStream],
    warning_messages: list[str],
) -> None:
    """N > K: the cap is reported in exactly one warning, however many bindings there are."""
    open_binding_stream(explicit_solution_source, {"solution_archive": OVER_CAP_COUNT})

    assert len(warning_messages) == 1


def test_generated_strategy_with_all_samples_never_receives_archive_rows(
    explicit_solution_source: SourceSpec,
    open_binding_stream: Callable[[SourceSpec, dict[str, int]], BatchStream],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gaussian_forward`` with ``samples=-1`` is handed no archive, despite the solution file."""
    from neuralls.domain.generation import strategy_rows

    real_run_generation = strategy_rows.run_generation
    archives_seen: list[object] = []

    def _spy(strategy_name: str, A: np.ndarray, *args: object, **kwargs: object) -> object:
        archives_seen.append(kwargs.get("archive"))
        return real_run_generation(strategy_name, A, *args, **kwargs)

    monkeypatch.setattr(strategy_rows, "run_generation", _spy)

    stream = open_binding_stream(explicit_solution_source, {"gaussian_forward": ALL_SAMPLES})
    assert list(stream.batches)

    assert archives_seen
    assert all(archive is None for archive in archives_seen)
