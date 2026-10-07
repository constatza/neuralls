"""Fixtures for identity derivation tests: on-disk job and data-profile TOML files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation import dense_streaming
from neuralls.domain.generation.batch import SampleBatch
from neuralls.domain.generation.sample_writer import SampleWriter
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.shared.types import DatasetFormat

type JobWriter = Callable[..., Path]

_JOB = """\
# a comment that must never matter
[run]
type = "fit"
seed = 42
precision = "64"
data = "{profile}"

{experiment}
[model]
name = "PODCoarseningFittable"
module_path = "neuralls.composition.preconditioners.pod_fittable"
rank = {rank}
"""

_PROFILE = """\
[data]
name = "FlexibleDataset"
batch_size = {batch_size}
pin_memory = true
shuffle = true

[data.module]
name = "ArrayDataModule"
"""


@pytest.fixture
def write_job(tmp_path: Path) -> JobWriter:
    """Write a fit-job TOML plus the data-profile TOML it references."""

    def _write(
        *,
        root: Path | None = None,
        rank: int = 10,
        batch_size: int = 256,
        experiment_name: str | None = None,
        comment: str = "",
    ) -> Path:
        base = root or tmp_path / "case"
        (base / "jobs").mkdir(parents=True, exist_ok=True)
        (base / "profiles").mkdir(parents=True, exist_ok=True)
        (base / "profiles" / "data.toml").write_text(_PROFILE.format(batch_size=batch_size))
        experiment = f'[experiment]\nname = "{experiment_name}"\n' if experiment_name else ""
        job = base / "jobs" / "job.toml"
        job.write_text(
            comment + _JOB.format(profile="../profiles/data.toml", experiment=experiment, rank=rank)
        )
        return job

    return _write


@pytest.fixture
def spd_system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A tiny symmetric positive-definite system: ``(A, b, x0)`` as NumPy arrays."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    initial_guess = np.zeros(2)
    return matrix, rhs, initial_guess


_ATOMIC_SEED = 2024
_ATOMIC_MATRIX_COUNT = 3
_ATOMIC_MATRIX_SIZE = 4
_ATOMIC_BATCH_ROWS = 4
"""Small batch size so the streamed run spans several batches."""


@pytest.fixture
def atomic_matrix_dir(tmp_path: Path) -> Path:
    """Seeded SPD matrices as individually globbable .txt files."""
    rng = np.random.default_rng(_ATOMIC_SEED)
    matrix_dir = tmp_path / "atomic_matrices"
    matrix_dir.mkdir()
    for index in range(_ATOMIC_MATRIX_COUNT):
        base = rng.standard_normal((_ATOMIC_MATRIX_SIZE, _ATOMIC_MATRIX_SIZE))
        matrix = base @ base.T + 5.0 * np.eye(_ATOMIC_MATRIX_SIZE)
        np.savetxt(matrix_dir / f"A_{index:03d}.txt", matrix)
    return matrix_dir


@pytest.fixture
def atomic_source(atomic_matrix_dir: Path) -> SourceSpec:
    return SourceSpec(matrix_path=str(atomic_matrix_dir / "A_*.txt"))


@pytest.fixture
def atomic_spec() -> DatasetSpec:
    """Dense streamed spec: forward-only strategy, several batches, fixed seed."""
    return DatasetSpec(
        mixture=MixtureSpec(
            counts={"gaussian_forward": 8},
            seed=_ATOMIC_SEED,
            shuffle=False,
        ),
        normalize="matrix",
        write_batch_size=_ATOMIC_BATCH_ROWS,
    )


@pytest.fixture(params=["zarr", "hdf5"])
def dense_format(request: pytest.FixtureRequest) -> DatasetFormat:
    return request.param


@pytest.fixture
def interrupt_writer(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Make the streamed writer raise after ``n`` batches have been written."""

    def _install(after_batches: int) -> None:
        written = 0

        class _InterruptedWriter(SampleWriter):
            def write_batch(self, batch: SampleBatch) -> None:
                nonlocal written
                if written >= after_batches:
                    raise RuntimeError("injected interruption")
                written += 1
                super().write_batch(batch)

        monkeypatch.setattr(dense_streaming, "SampleWriter", _InterruptedWriter)

    return _install
