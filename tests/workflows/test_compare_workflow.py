from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from neuralls.composition.comparison import comparison_run
from neuralls.composition.comparison.comparison_run import (
    compare_preconditioners,
)
from neuralls.composition.comparison.models import LinearSystem
from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.result import CGComparisonResult, PlotPaths
from neuralls.platform.config.loaders import load_comparison_config
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.storage.datasets import DenseDatasetWriter, DenseZarrAccumulator


def _assert_under(path: Path, root: Path) -> None:
    assert path.resolve().is_relative_to(root.resolve())


def _write_dataset(root: Path, A: np.ndarray, b: np.ndarray) -> None:
    solutions = np.linalg.solve(A, b)
    rhs = b.reshape(1, -1)
    sols = solutions.reshape(1, -1)
    acc = DenseZarrAccumulator(root / "matrix.zarr")
    acc.append_dense_matrix(A, repeats=1)
    zarr_path = acc.finalize()
    payload = GeneratedDatasetPayload(
        rhs=rhs,
        solutions=sols,
        matrix_artifact_path=zarr_path,
        matrix_size=(int(A.shape[0]), int(A.shape[1])),
        normalization_type="matrix",
        matrix_norm=float(np.linalg.norm(A, ord=2)),
        matrix_norm_type="spectral",
    )
    DenseDatasetWriter().write_dataset(root, payload)


def _write_comparison_config(path: Path, system_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[general]",
                "",
                "[general.params]",
                "rtol = 1e-6",
                "atol = 1e-14",
                "max_iterations = 50",
                'stopping_criterion = "residual_norm"',
                "",
                "[general.data]",
                f'matrix_path = "{system_path.as_posix()}"',
                f'rhs_path = "{system_path.as_posix()}"',
                'normalize_system = "matrix"',
                "",
                "[[preconditioners]]",
                'name = "none"',
                'type = "identity"',
                "",
                "[[preconditioners]]",
                'name = "jacobi"',
                'type = "jacobi"',
            ]
        ),
        encoding="utf-8",
    )


def _write_data_config(path: Path, data_root: Path) -> None:
    """Write minimal data config for test."""
    path.write_text(
        "\n".join(
            [
                'id = "test-dataset"',
                "",
                "[output]",
                f'data_dir = "{data_root.as_posix()}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_model_config(path: Path, checkpoint_dir: Path) -> None:
    """Write minimal model config for test."""
    profile_path = path.with_name(f"{path.stem}-profile.toml")
    profile_path.write_text(
        '[model]\nname = "ScaleEquivariantFFNN"\n\n[data]\nname = "FlexibleDataset"\n\n[data.module]\nname = "ArrayDataModule"',
        encoding="utf-8",
    )
    path.write_text(
        "\n".join(
            [
                "[run]",
                'type = "train"',
                "seed = 42",
                f'model = "{profile_path.name}"',
                f'data = "{profile_path.name}"',
                "",
                "[experiment]",
                'name = "test-experiment"',
                "",
                "[training.trainer]",
                "max_epochs = 1",
                "",
                "[training.optimizer.default_optimizer]",
                'name = "AdamW"',
                "lr = 0.001",
            ]
        ),
        encoding="utf-8",
    )


def _cg_result(name: str) -> CGComparisonResult:
    return CGComparisonResult(
        x=np.array([1.0, 2.0]),
        converged=True,
        iterations=1,
        residual=0.0,
        residual_abs=0.0,
        residual_history_rel=[1.0, 0.0],
        residual_history_abs=[1.0, 0.0],
        preconditioner=name,
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
    )


def test_compare_preconditioners_evaluates_configs_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison orchestration must not construct all live preconditioners up front."""
    events: list[str] = []
    active: set[str] = set()

    class TrackedPreconditioner:
        def __init__(self, name: str) -> None:
            if active:
                raise AssertionError(f"constructed {name} while {active} still active")
            self.name = name
            active.add(name)
            events.append(f"create:{name}")

        def cleanup(self) -> None:
            active.discard(self.name)
            events.append(f"cleanup:{self.name}")

    class TrackedService:
        def create_preconditioner(
            self,
            matrix: torch.Tensor,
            config: StandardPreconditionerConfig,
        ) -> TrackedPreconditioner:
            del matrix
            return TrackedPreconditioner(config.name)

    def fake_condition_numbers(
        matrix: np.ndarray,
        preconditioners: dict[str, TrackedPreconditioner],
    ) -> dict[str, float]:
        del matrix
        assert set(preconditioners) == set(active)
        return {name: 1.0 for name in preconditioners}

    def fake_run_cg_comparison(
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        *,
        preconditioners: dict[str, TrackedPreconditioner],
        **kwargs: Any,
    ) -> dict[str, CGComparisonResult]:
        del matrix, rhs, kwargs
        assert set(preconditioners) == set(active)
        name = next(iter(preconditioners))
        result = {name: _cg_result(name)}
        preconditioners[name].cleanup()
        return result

    monkeypatch.setattr(
        comparison_run,
        "_load_linear_system",
        lambda *args, **kwargs: LinearSystem(
            matrix=torch.eye(2, dtype=torch.float64),
            rhs=torch.ones(2, dtype=torch.float64),
        ),
    )
    monkeypatch.setattr(comparison_run, "_ensure_comparison_directories", lambda paths: None)
    monkeypatch.setattr(
        comparison_run, "_log_matrix_condition_number", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(comparison_run, "PreconditionerService", TrackedService)
    monkeypatch.setattr(comparison_run, "compute_condition_numbers", fake_condition_numbers)
    monkeypatch.setattr(
        comparison_run,
        "build_preconditioner_labels",
        lambda preconditioners: {name: name for name in preconditioners},
    )
    monkeypatch.setattr(comparison_run, "run_cg_comparison", fake_run_cg_comparison)
    monkeypatch.setattr(
        comparison_run,
        "_generate_comparison_plots",
        lambda *args, **kwargs: PlotPaths(),
    )

    general = ComparisonGeneral(
        params=SolverParams(
            rtol=1.0e-6,
            atol=1.0e-14,
            max_iterations=2,
            stopping_criterion="residual_norm",
            m_max=2,
        ),
        data=ComparisonData(
            matrix_path=Path("unused-matrix.npy"),
            rhs_path=Path("unused-rhs.npy"),
        ),
    )
    specs = (
        StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(name="second", type=PreconditionerType.JACOBI),
    )

    compare_preconditioners(
        general_params=general,
        preconditioner_configs=specs,
        output_root=Path("unused-output"),
    )

    assert events == ["create:none", "cleanup:none", "create:second", "cleanup:second"]
    assert not active


def test_compare_preconditioners_workflow(tmp_path: Path, neuralls_settings) -> None:
    """End-to-end check that compare_preconditioners loads config and data."""
    A = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    _write_dataset(tmp_path, A, b)

    comparison_cfg = tmp_path / "comparison_config.toml"
    _write_comparison_config(comparison_cfg, tmp_path)

    # Create minimal data config pointing to tmp_path
    data_cfg = tmp_path / "data_config.toml"
    _write_data_config(data_cfg, tmp_path)

    # Create minimal model config
    model_cfg = tmp_path / "model_config.toml"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    _write_model_config(model_cfg, checkpoint_dir)

    comparison_cfg_model = load_comparison_config(comparison_cfg, neuralls_settings)
    output_root = tmp_path / "comparison-output"
    results = compare_preconditioners(
        general_params=comparison_cfg_model.general,
        preconditioner_configs=comparison_cfg_model.preconditioners,
        output_root=output_root,
    )

    # Access typed solver results from ComparisonResult
    comparison_results = results.results
    assert set(comparison_results.keys()) == {"none", "jacobi"}
    for name, info in comparison_results.items():
        assert info.iterations > 0, f"{name} did not run"
    _assert_under(results.output_dir, tmp_path)
    for plot_path in results.plot_paths.to_mapping().values():
        assert plot_path.is_file()
        _assert_under(plot_path, tmp_path)
