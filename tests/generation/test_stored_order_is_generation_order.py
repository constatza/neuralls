"""Stored order equals generation order for every dataset.

The write-time shuffle was dropped: dlkit shuffles the training loader every epoch,
so the stored order carries no training signal and is kept as the order in which the
strategies emitted their rows. These tests pin that contract on a two-binding
mixture, where binding order and strategy order are both observable in the stored
arrays.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from loguru import logger

from neuralls.composition.generation._context_builder import _warn_if_shuffle_requested
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.platform.config.models.data_models import GenerationConfig
from neuralls.platform.storage.dataset_readers import (
    load_matrix_sample_index,
    load_row_kind_codes,
)
from neuralls.platform.storage.datasets import load_dense_training_arrays

_SEED = 1234
_MATRIX_COUNT = 2
_MATRIX_SIZE = 5
_COUNTS = {"gaussian_forward": 6, "gaussian_residuals": 6}
_STRATEGY_OVERRIDES = {"gaussian_residuals": {"stop": 3, "start": 1}}


@pytest.fixture
def two_matrix_dir(tmp_path: Path) -> Path:
    """Two deterministic SPD matrices as individually globbable .txt files."""
    rng = np.random.default_rng(7)
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    for index in range(_MATRIX_COUNT):
        base = rng.standard_normal((_MATRIX_SIZE, _MATRIX_SIZE))
        matrix = base @ base.T + 5.0 * np.eye(_MATRIX_SIZE)
        np.savetxt(matrix_dir / f"A_{index:03d}.txt", matrix)
    return matrix_dir


def _build(
    matrix_dir: Path,
    out_dir: Path,
    solver_overrides: dict[str, Any],
    **mixture_kwargs: Any,
) -> Path:
    build_dataset(
        SourceSpec(matrix_path=str(matrix_dir / "A_*.txt")),
        DatasetSpec(
            mixture=MixtureSpec(
                counts=_COUNTS,
                seed=_SEED,
                strategy_overrides=_STRATEGY_OVERRIDES,
                solver_overrides=solver_overrides,
                **mixture_kwargs,
            ),
            normalize="matrix",
        ),
        str(out_dir),
        dataset_format="hdf5",
    )
    return out_dir


@pytest.fixture
def unshuffled_dataset(
    two_matrix_dir: Path, tmp_path: Path, solver_overrides: dict[str, Any]
) -> Path:
    return _build(two_matrix_dir, tmp_path / "unshuffled", solver_overrides, shuffle=False)


@pytest.fixture
def default_dataset(two_matrix_dir: Path, tmp_path: Path, solver_overrides: dict[str, Any]) -> Path:
    return _build(two_matrix_dir, tmp_path / "default", solver_overrides)


def test_default_stored_order_matches_unshuffled_reference(
    unshuffled_dataset: Path, default_dataset: Path
) -> None:
    ref_rhs, ref_sol = load_dense_training_arrays(unshuffled_dataset)
    rhs, sol = load_dense_training_arrays(default_dataset)

    np.testing.assert_array_equal(rhs, ref_rhs)
    np.testing.assert_array_equal(sol, ref_sol)
    np.testing.assert_array_equal(
        load_row_kind_codes(default_dataset), load_row_kind_codes(unshuffled_dataset)
    )
    np.testing.assert_array_equal(
        load_matrix_sample_index(default_dataset),
        load_matrix_sample_index(unshuffled_dataset),
    )


def test_bindings_are_stored_in_binding_order(default_dataset: Path) -> None:
    matrix_index = load_matrix_sample_index(default_dataset)
    assert np.all(np.diff(matrix_index) >= 0)
    assert set(np.unique(matrix_index).tolist()) == set(range(_MATRIX_COUNT))


def test_strategy_order_is_preserved_within_each_binding(
    unshuffled_dataset: Path, default_dataset: Path
) -> None:
    row_kinds = load_row_kind_codes(default_dataset)
    matrix_index = load_matrix_sample_index(default_dataset)
    for binding in range(_MATRIX_COUNT):
        block = row_kinds[matrix_index == binding]
        assert block.size > 0
        # Strategy blocks are contiguous and emitted in strategy order, so the
        # codes change value at most once per strategy boundary.
        boundaries = np.flatnonzero(np.diff(block) != 0)
        assert len(boundaries) <= len(_COUNTS) - 1
        assert np.all(np.diff(block) >= 0) or np.all(np.diff(block) <= 0)


@pytest.fixture
def captured_warnings() -> Iterator[list[str]]:
    """Collects loguru WARNING messages for the duration of one test."""
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    yield messages
    logger.remove(sink_id)


def test_explicit_shuffle_true_logs_one_warning(captured_warnings: list[str]) -> None:
    _warn_if_shuffle_requested(GenerationConfig.model_validate({"shuffle": True}))
    assert len([m for m in captured_warnings if "generation order" in m]) == 1


def test_default_shuffle_does_not_warn(captured_warnings: list[str]) -> None:
    _warn_if_shuffle_requested(GenerationConfig())
    assert captured_warnings == []
