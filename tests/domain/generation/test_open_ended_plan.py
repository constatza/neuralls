"""Open-ended (samples = -1) archive counts must be sized before generation.

The archive pool is the glob's file count, which is a directory listing, so a
``samples = -1`` archive strategy over several matrices has a known row total
without reading any file contents.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.batch_plan import ALL_SAMPLES, plan_batches
from neuralls.domain.generation.binding_allocation import (
    _archive_pool_size,
    _resolve_binding_strategy_counts,
)
from neuralls.domain.generation.bindings import SystemBinding
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec

_NUM_MATRICES = 3
_NUM_FILES = 5
_ARCHIVE_STRATEGY = "rhs_archive"
_GLOB_KEY = "rhs_glob"
_FILE_SEED = 7
_FILE_LENGTH = 4


@pytest.fixture
def archive_dir(tmp_path: Path) -> Path:
    """Directory holding ``_NUM_FILES`` small seeded vector files."""
    directory = tmp_path / "archive"
    directory.mkdir()
    rng = np.random.default_rng(_FILE_SEED)
    for file_idx in range(_NUM_FILES):
        vector = rng.standard_normal(_FILE_LENGTH)
        np.savetxt(directory / f"rhs_{file_idx:02d}.txt", vector)
    return directory


@pytest.fixture
def archive_glob(archive_dir: Path) -> str:
    """Glob matching every archive file in ``archive_dir``."""
    return str(archive_dir / "rhs_*.txt")


def _open_ended_spec(strategy: str, glob: str) -> DatasetSpec:
    """A single open-ended archive strategy reading the given glob."""
    return DatasetSpec(
        mixture=MixtureSpec(
            counts={strategy: ALL_SAMPLES},
            seed=0,
            strategy_overrides={strategy: {_GLOB_KEY: glob}},
        ),
        normalize="matrix",
    )


def test_all_samples_archive_total_is_known_before_generation(
    archive_glob: str,
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    spec = _open_ended_spec(_ARCHIVE_STRATEGY, archive_glob)
    bindings = make_bindings(_NUM_MATRICES, 1)

    allocation = _resolve_binding_strategy_counts(
        bindings=bindings,
        spec=spec,
        num_matrix_samples=_NUM_MATRICES,
    )
    plan = plan_batches(allocation)

    assert plan.is_exact
    assert plan.total_rows == _NUM_MATRICES * _NUM_FILES


def test_all_samples_per_strategy_counts_match_the_cyclic_map(
    archive_glob: str,
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    spec = _open_ended_spec(_ARCHIVE_STRATEGY, archive_glob)
    bindings = make_bindings(_NUM_MATRICES, 1)

    allocation = _resolve_binding_strategy_counts(
        bindings=bindings,
        spec=spec,
        num_matrix_samples=_NUM_MATRICES,
    )

    # Unit t maps to matrix t % M and file (matrix + t // M) % K, so matrix m walks the
    # pool starting at file m. Every matrix ends up with all K files, each once.
    assert _archive_pool_size(archive_glob, skip=0) == _NUM_FILES
    expected_files = [
        tuple((matrix + pass_idx) % _NUM_FILES for pass_idx in range(_NUM_FILES))
        for matrix in range(_NUM_MATRICES)
    ]
    assert [binding_counts[_ARCHIVE_STRATEGY] for binding_counts in allocation.counts] == [
        _NUM_FILES
    ] * _NUM_MATRICES
    assert [binding_files[_ARCHIVE_STRATEGY] for binding_files in allocation.file_indices] == (
        expected_files
    )


def test_single_matrix_all_samples_archive_is_sized(
    archive_glob: str,
    make_bindings: Callable[[int, int], list[SystemBinding]],
) -> None:
    # One matrix means one pool of K files, so the cyclic map yields K units in glob order.
    spec = _open_ended_spec(_ARCHIVE_STRATEGY, archive_glob)
    bindings = make_bindings(1, 1)

    allocation = _resolve_binding_strategy_counts(
        bindings=bindings,
        spec=spec,
        num_matrix_samples=1,
    )
    plan = plan_batches(allocation)

    assert plan.is_exact
    assert plan.total_rows == _NUM_FILES
    assert allocation.file_indices[0][_ARCHIVE_STRATEGY] == tuple(range(_NUM_FILES))
