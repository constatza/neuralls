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
from neuralls.composition.comparison.result_keys import validate_unique_preconditioner_keys
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.result import CGComparisonResult, PlotPaths
from neuralls.platform.config.loaders import load_comparison_config
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)


def _assert_under(path: Path, root: Path) -> None:
    assert path.resolve().is_relative_to(root.resolve())


def _write_dataset(root: Path, A: np.ndarray) -> None:
    """Stream a one-row generated dataset for ``A`` into ``root`` (hdf5, no zarr in tests)."""
    matrix_path = root / "matrix.npy"
    np.save(matrix_path, A)
    build_dataset(
        SourceSpec(matrix_path=str(matrix_path)),
        DatasetSpec(
            mixture=MixtureSpec(counts={"gaussian_forward": 1}, seed=0, shuffle=False),
            normalize="none",
        ),
        str(root),
        dataset_format="hdf5",
    )


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
                'name = "identity"',
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


def test_preconditioner_result_keys_reject_duplicate_keys() -> None:
    """Duplicate wiring keys cannot silently collapse two solver cases."""
    specs = (
        StandardPreconditionerConfig(name="same", type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(name="same", type=PreconditionerType.JACOBI),
    )

    with pytest.raises(ValueError, match="duplicate keys: 'same'"):
        validate_unique_preconditioner_keys(specs)


def test_preconditioner_result_keys_do_not_filter_by_preconditioner_type(
    same_type_distinct_key_specs: tuple[StandardPreconditionerConfig, ...],
) -> None:
    """Distinct result ids remain valid even when their typed implementations match."""
    assert validate_unique_preconditioner_keys(same_type_distinct_key_specs) == (
        "pcg-identity",
        "future-solver-identity",
    )


def test_compare_preconditioners_evaluates_configs_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
    neuralls_settings,
) -> None:
    """Comparison evaluates one preconditioner at a time on the solver device."""
    events: list[str] = []
    active: set[str] = set()
    expected_device = torch.device("meta")

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
            assert matrix.device == expected_device
            return TrackedPreconditioner(config.name)

    def fake_run_cg_comparison(
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        *,
        preconditioners: dict[str, TrackedPreconditioner],
        **kwargs: Any,
    ) -> dict[str, CGComparisonResult]:
        del kwargs
        assert matrix.device == expected_device
        assert rhs.device == expected_device
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
    monkeypatch.setattr(comparison_run, "resolve_device", lambda: expected_device)
    monkeypatch.setattr(comparison_run, "compute_reference_solution", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparison_run, "PreconditionerService", TrackedService)
    monkeypatch.setattr(
        comparison_run,
        "build_preconditioner_labels",
        lambda preconditioners, families: {name: name for name in preconditioners},
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
        StandardPreconditionerConfig(name="first", type=PreconditionerType.JACOBI),
        StandardPreconditionerConfig(name="second", type=PreconditionerType.ILU),
    )

    compare_preconditioners(
        general_params=general,
        preconditioner_configs=specs,
        output_root=Path("unused-output"),
        settings=neuralls_settings,
    )

    assert events == ["create:first", "cleanup:first", "create:second", "cleanup:second"]
    assert not active


def test_compare_preconditioners_workflow(tmp_path: Path, neuralls_settings) -> None:
    """End-to-end check that compare_preconditioners loads config and data."""
    A = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
    _write_dataset(tmp_path, A)

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
        settings=neuralls_settings,
    )

    # Access typed solver results from ComparisonResult
    comparison_results = results.results
    assert set(comparison_results.keys()) == {"identity", "jacobi"}
    for name, info in comparison_results.items():
        assert info.iterations > 0, f"{name} did not run"
        assert info.setup_cost >= 0
        assert info.solve_time_seconds >= 0
        assert info.peak_memory_bytes >= 0
    _assert_under(results.output_dir, tmp_path)
    for plot_path in results.plot_paths.to_mapping().values():
        assert plot_path.is_file()
        _assert_under(plot_path, tmp_path)
