"""Fixtures for identity-marker tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.data_models import (
    DataConfigFile,
    GenerationConfig,
    OutputConfig,
    SourceConfig,
    StrategyConfig,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    IC0PreconditionerConfig,
    PODCoarseningConfig,
    StandardPreconditionerConfig,
)

type ComparisonFactory = Callable[..., ComparisonConfig]
type DataConfigFactory = Callable[..., DataConfigFile]


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    """Generated-dataset directory feeding the POD coarsening config."""
    root = tmp_path / "snapshots"
    root.mkdir()
    (root / "solutions.bin").write_bytes(b"\x00\x01\x02")
    return root


@pytest.fixture
def matrix_file(tmp_path: Path) -> Path:
    """Matrix file location used by the comparison data spec."""
    path = tmp_path / "matrix.bin"
    path.write_bytes(b"matrix")
    return path


@pytest.fixture
def make_comparison(matrix_file: Path, snapshot_dir: Path) -> ComparisonFactory:
    """Build a ComparisonConfig, overriding selected fields."""

    def _make(
        *,
        rtol: float = 1e-8,
        max_iterations: int = 100,
        ic0_threshold: float = 0.0,
        pod_rank: int = 4,
        rhs_std: float = 1.0,
        jacobi_name: str = "jacobi",
        matrix_path: Path | None = None,
    ) -> ComparisonConfig:
        data = ComparisonData(
            matrix_path=matrix_path or matrix_file,
            rhs_path=None,
            rhs_source_kind=None,
            rhs_source_params={"kind": "gaussian", "std": rhs_std},
        )
        params = SolverParams(
            rtol=rtol,
            atol=0.0,
            max_iterations=max_iterations,
            stopping_criterion="residual_norm",
            m_max=1,
        )
        return ComparisonConfig(
            general=ComparisonGeneral(params=params, data=data),
            preconditioners=(
                StandardPreconditionerConfig(type="jacobi", name=jacobi_name),
                IC0PreconditionerConfig(threshold=ic0_threshold),
                AMGPreconditionerConfig(
                    coarsening=PODCoarseningConfig(dataset_dir=snapshot_dir, rank=pod_rank)
                ),
            ),
        )

    return _make


@pytest.fixture
def make_data_config(tmp_path: Path) -> DataConfigFactory:
    """Build a DataConfigFile, overriding selected fields."""

    def _make(
        *, dataset_id: str = "ds", data_dir: Path | None = None, seed: int = 42, samples: int = 8
    ) -> DataConfigFile:
        return DataConfigFile(
            id=dataset_id,
            source=SourceConfig(type="generated", matrix_path=str(tmp_path / "m.mtx")),
            generation=GenerationConfig(
                seed=seed, strategy=[StrategyConfig(name="gaussian_residuals", samples=samples)]
            ),
            output=OutputConfig(data_dir=data_dir or tmp_path / "out"),
        )

    return _make
