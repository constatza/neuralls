"""Shared fixtures for generation-domain allocation tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from neuralls.domain.generation.source_streams import SystemBinding
from neuralls.domain.generation.specs import MixtureSpec


@pytest.fixture
def make_bindings() -> Callable[[int, int], list[SystemBinding]]:
    """Build bindings for ``num_matrices`` matrices with ``per_matrix`` bindings each.

    Matrix ``m`` owns the consecutive binding positions it is given, so the
    returned list is ordered by matrix.
    """

    def _build(num_matrices: int, per_matrix: int) -> list[SystemBinding]:
        return [
            SystemBinding(sample_id=m * per_matrix + k, matrix_sample_id=m)
            for m in range(num_matrices)
            for k in range(per_matrix)
        ]

    return _build


@pytest.fixture
def make_mixture() -> Callable[..., MixtureSpec]:
    """Build a single-strategy ``MixtureSpec``, optionally overriding the solutions glob."""

    def _build(strategy: str, count: int, *, glob: str | None = None) -> MixtureSpec:
        overrides = {strategy: {"solutions_glob": glob}} if glob is not None else None
        return MixtureSpec(counts={strategy: count}, seed=0, strategy_overrides=overrides)

    return _build


@pytest.fixture
def make_archive_glob(tmp_path: Path) -> Callable[[int], str]:
    """Create ``num_files`` archive files under ``tmp_path`` and return their glob.

    Only the file names matter to the allocator, which lists the pool before any read.
    """

    def _build(num_files: int) -> str:
        directory = tmp_path / "archive"
        directory.mkdir(exist_ok=True)
        for k in range(num_files):
            (directory / f"solution_{k:03d}.txt").write_text("0.0\n")
        return str(directory / "solution_*.txt")

    return _build


TRACE_WINDOW_OVERRIDES: dict[str, int] = {"start": 0, "stop": 3, "step": 1}
"""Window giving K_rows = 4 (rows 0..3 of a trajectory run to stop=3)."""
