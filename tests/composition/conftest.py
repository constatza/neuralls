"""Fixtures for identity derivation tests: on-disk job and data-profile TOML files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

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
