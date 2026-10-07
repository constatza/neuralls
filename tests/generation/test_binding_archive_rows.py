"""Explicit solution files feed archive strategies only, and only by their declared rule.

Cyclic counts draw rows ``(b + p) mod K`` for binding ``b``; ``samples=-1`` takes every row.
A generated strategy never receives archive rows, whatever its count.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.orchestration import BatchStream
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.datasets import load_dense_training_arrays

ALL_SAMPLES = -1
CYCLIC_COUNT = 2
OVER_CAP_COUNT = 6


@pytest.fixture
def explicit_solution_source(three_spd_matrix_dir: Path, solution_block_npy: Path) -> SourceSpec:
    """Three matrices, each binding paired with the one explicit solution file."""
    return SourceSpec(
        matrix_path=str(three_spd_matrix_dir / "A_*.txt"),
        solution_path=str(solution_block_npy),
        sample_id_regex=r"(\d+)(?!.*\d)",
    )


def _binding_solutions(stream: BatchStream, n_bindings: int) -> list[np.ndarray]:
    """Solution rows of each binding, concatenated over that binding's batches in order."""
    batches = list(stream.batches)
    return [
        np.concatenate([b.solutions for b in batches if b.binding_index == index])
        for index in range(n_bindings)
    ]


def test_cyclic_count_draws_rows_mod_k_per_binding(
    explicit_solution_source: SourceSpec,
    open_binding_stream: Callable[..., BatchStream],
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
    open_binding_stream: Callable[..., BatchStream],
    warning_messages: list[str],
) -> None:
    """N > K: the cap is reported in exactly one warning, however many bindings there are."""
    open_binding_stream(explicit_solution_source, {"solution_archive": OVER_CAP_COUNT})

    capped = [message for message in warning_messages if str(OVER_CAP_COUNT) in message]
    assert len(capped) == 1


@pytest.fixture
def build_solution_dataset(
    tmp_path: Path,
) -> Callable[[SourceSpec, dict[str, int]], tuple[np.ndarray, np.ndarray]]:
    """Build a dataset from a source and counts, returning its stored ``(rhs, solutions)``."""

    def _build(source: SourceSpec, counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
        dataset_dir = str(tmp_path / "built")
        build_dataset(source, DatasetSpec(mixture=MixtureSpec(counts=counts, seed=7)), dataset_dir)
        return load_dense_training_arrays(dataset_dir)

    return _build


def test_all_samples_plan_total_equals_written_total(
    explicit_solution_source: SourceSpec,
    open_binding_stream: Callable[..., BatchStream],
    build_solution_dataset: Callable[[SourceSpec, dict[str, int]], tuple[np.ndarray, np.ndarray]],
) -> None:
    """``samples=-1`` over 3 bindings: the plan total is the number of rows actually written."""
    counts = {"solution_archive": ALL_SAMPLES}
    planned = open_binding_stream(explicit_solution_source, counts).plan.total_rows

    _, solutions = build_solution_dataset(explicit_solution_source, counts)

    assert planned == solutions.shape[0]


def test_generated_strategy_with_all_samples_never_receives_archive_rows(
    explicit_solution_source: SourceSpec,
    open_binding_stream: Callable[..., BatchStream],
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


def test_single_matrix_glob_solution_with_all_samples_writes_every_file(
    three_spd_matrix_dir: Path,
    tmp_path: Path,
    solution_block: np.ndarray,
    open_binding_stream: Callable[..., BatchStream],
    build_solution_dataset: Callable[[SourceSpec, dict[str, int]], tuple[np.ndarray, np.ndarray]],
) -> None:
    """One matrix with a glob of K solution files and ``samples=-1``: plan and output both hold K rows."""
    solution_dir = tmp_path / "solution_files"
    solution_dir.mkdir()
    for i, vector in enumerate(solution_block):
        np.savetxt(solution_dir / f"sol_{i}.txt", vector)
    source = SourceSpec(
        matrix_path=str(three_spd_matrix_dir / "A_0.txt"),
        solution_path=str(solution_dir / "sol_*.txt"),
        sample_id_regex=r"(\d+)(?!.*\d)",
    )
    counts = {"solution_archive": ALL_SAMPLES}

    planned = open_binding_stream(source, counts).plan.total_rows
    _, solutions = build_solution_dataset(source, counts)

    assert planned == solution_block.shape[0]
    assert solutions.shape[0] == solution_block.shape[0]
