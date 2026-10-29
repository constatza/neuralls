"""Fixtures for identity derivation tests: on-disk job and data-profile TOML files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.presentation.provenance import DatasetSourceProvenance
from neuralls.shared.scientific_description.models import ScientificDescription
from tests import atomic_support
from tests.dataset_support import build_test_dataset, strip_provenance

type JobWriter = Callable[..., Path]

# Re-exported so these tests share the seeded dense-generation fixtures with the other conftest.
atomic_matrix_dir = atomic_support.atomic_matrix_dir
atomic_source = atomic_support.atomic_source
atomic_spec = atomic_support.atomic_spec

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
        (base / "profiles" / "data.toml").write_text(
            _PROFILE.format(batch_size=batch_size), encoding="utf-8"
        )
        experiment = f'[experiment]\nname = "{experiment_name}"\n' if experiment_name else ""
        job = base / "jobs" / "job.toml"
        job.write_text(
            comment
            + _JOB.format(profile="../profiles/data.toml", experiment=experiment, rank=rank),
            encoding="utf-8",
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


@pytest.fixture
def pod_dataset_dir(tmp_path: Path, atomic_source: SourceSpec, atomic_spec: DatasetSpec) -> Path:
    """A real generated dataset whose manifest carries scientific provenance."""
    target = tmp_path / "pod-fit"
    build_test_dataset(
        atomic_source,
        atomic_spec,
        str(target),
        source_provenance=DatasetSourceProvenance(ScientificDescription("Atomic test")),
    )
    return target


@pytest.fixture
def stale_dataset_dir(tmp_path: Path, atomic_source: SourceSpec, atomic_spec: DatasetSpec) -> Path:
    """A generated dataset whose manifest predates provenance."""
    target = tmp_path / "stale"
    build_test_dataset(atomic_source, atomic_spec, str(target))
    strip_provenance(target)
    return target
