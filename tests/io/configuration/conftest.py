"""Fixtures for sweep-expansion config-loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_SWEEP_SPEC = """\
id = "sweep-ds"

[source]
matrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"

[generation]
seed = 1

[[generation.strategy]]
name = "gaussian_residuals"
samples = [10, 20]
stop = 5

[output]
data_dir = "${NEURALLS_PROCESSED_DIR}"
"""


@pytest.fixture
def sweep_dataset_dir(tmp_path: Path) -> Path:
    """Directory holding one `gaussian-cg.sweep.toml` (samples 10 and 20)."""
    directory = tmp_path / "datasets"
    directory.mkdir()
    (directory / "gaussian-cg.sweep.toml").write_text(_SWEEP_SPEC, encoding="utf-8")
    return directory


@pytest.fixture
def generated_path(sweep_dataset_dir: Path) -> Path:
    """The `_generated/` file the sweep fixture produces for samples = 10."""
    return sweep_dataset_dir / "_generated" / "gaussian-cg-10.toml"


@pytest.fixture
def sweep_case_config(tmp_path: Path, generated_path: Path) -> Path:
    """Case config whose only dataset is the not-yet-generated sweep output."""
    config = tmp_path / "case.toml"
    config.write_text(f'[[datasets]]\npath = "{generated_path.as_posix()}"\n', encoding="utf-8")
    return config
