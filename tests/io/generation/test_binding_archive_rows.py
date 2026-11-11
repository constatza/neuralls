"""``samples=-1`` over explicit solution files plans exactly the rows the build writes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from tests.dataset_support import build_test_dataset

from neuralls.domain.generation.batch_plan import ALL_SAMPLES
from neuralls.domain.generation.orchestration import BatchStream
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.storage.datasets import load_dense_training_arrays


@pytest.fixture
def explicit_solution_source(three_spd_matrix_dir: Path, solution_block_npy: Path) -> SourceSpec:
    """Three matrices, each binding paired with the one explicit solution file."""
    return SourceSpec(
        matrix_path=str(three_spd_matrix_dir / "A_*.txt"),
        solution_path=str(solution_block_npy),
        sample_id_regex=r"(\d+)(?!.*\d)",
    )


@pytest.fixture
def build_solution_dataset(
    tmp_path: Path,
) -> Callable[[SourceSpec, dict[str, int]], tuple[np.ndarray, np.ndarray]]:
    """Build a dataset from a source and counts, returning its stored ``(rhs, solutions)``."""

    def _build(source: SourceSpec, counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
        dataset_dir = str(tmp_path / "built")
        build_test_dataset(
            source, DatasetSpec(mixture=MixtureSpec(counts=counts, seed=7)), dataset_dir
        )
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
