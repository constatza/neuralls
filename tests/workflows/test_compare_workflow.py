from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from neuralls.composition.assignments.presentation_preflight import prepare_comparison_presentation
from neuralls.composition.comparison import comparison_run
from neuralls.composition.comparison.comparison_run import (
    compare_preconditioners,
)
from neuralls.composition.comparison.models import ComparisonPresentation, LinearSystem
from neuralls.composition.comparison.result_keys import validate_unique_preconditioner_keys
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.plot_paths import PlotPaths
from neuralls.platform.config.loaders import load_comparison_config
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting.figure_text import method_legend
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from neuralls.shared.types import PreconditionerName
from tests.dataset_support import build_test_dataset


def _assert_under(path: Path, root: Path) -> None:
    assert path.resolve().is_relative_to(root.resolve())


def _write_dataset(root: Path, A: np.ndarray) -> None:
    """Stream a one-row generated dataset for ``A`` into ``root`` (hdf5, no zarr in tests)."""
    matrix_path = root / "matrix.npy"
    np.save(matrix_path, A)
    build_test_dataset(
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
    """Two configs with identical type and fields derive the identical key and must be rejected.

    `name` is no longer user-settable — it's derived from type + fields (see
    `preconditioner.py::preconditioner_slug`), so a "duplicate key" now means
    what it should: two requested solves that are actually the same
    preconditioner, listed twice.
    """
    specs = (
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
    )

    with pytest.raises(ValueError):
        validate_unique_preconditioner_keys(specs)


def test_preconditioner_result_keys_differ_when_fields_differ() -> None:
    """Same type, different distinguishing field -> different derived keys, no collision."""
    specs = (
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI),
    )

    assert validate_unique_preconditioner_keys(specs) == ("identity", "jacobi")


def test_compare_preconditioners_evaluates_configs_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
    neuralls_settings,
    dataset_presentation: DatasetPresentation,
) -> None:
    """Comparison evaluates one preconditioner at a time on the solver device."""
    events: list[str] = []
    plot_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
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
        def __init__(self, *, checkpoints: object) -> None:
            del checkpoints

        def preload(self, matrix: torch.Tensor, config: StandardPreconditionerConfig) -> None:
            """No checkpoints in this test's preconditioners."""

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
    monkeypatch.setattr(comparison_run, "run_cg_comparison", fake_run_cg_comparison)
    monkeypatch.setattr(
        comparison_run,
        "_generate_comparison_plots",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)) or PlotPaths(),
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
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI),
        StandardPreconditionerConfig(type=PreconditionerType.ILU),
    )
    mathtext = NotationPolicy.default(RenderTarget.MATHTEXT)
    prepared = prepare_comparison_presentation(
        specs, ComparisonScopePresentation(dataset_presentation)
    )

    compare_preconditioners(
        general_params=general,
        preconditioner_configs=specs,
        output_root=Path("unused-output"),
        settings=neuralls_settings,
        presentation=ComparisonPresentation(
            methods=prepared,
            title=ComparisonScopePresentation(dataset_presentation),
            notation=mathtext,
        ),
    )

    assert events == ["create:jacobi", "cleanup:jacobi", "create:ilu", "cleanup:ilu"]
    assert not active
    # The prepared presentation, not the configs, names the methods in the figures.
    legend, title = plot_calls[0][0][2:4]
    assert legend == method_legend(
        {spec.name: prepared.method(PreconditionerName(spec.name)).compact for spec in specs},
        mathtext,
    )
    assert title.notation == mathtext


@pytest.mark.integration
def test_compare_preconditioners_workflow(
    tmp_path: Path, neuralls_settings, dataset_presentation: DatasetPresentation
) -> None:
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
        presentation=ComparisonPresentation(
            methods=prepare_comparison_presentation(
                comparison_cfg_model.preconditioners,
                ComparisonScopePresentation(dataset_presentation),
            ),
            title=ComparisonScopePresentation(dataset_presentation),
            notation=NotationPolicy.default(RenderTarget.MATHTEXT),
        ),
    )

    # Access typed solver results from ComparisonResult
    comparison_results = results.results
    assert set(comparison_results.keys()) == {"identity", "jacobi"}
    for name, info in comparison_results.items():
        assert info.iterations > 0, f"{name} did not run"
        assert info.setup_cost >= 0
        assert info.solve_time_seconds is not None
        assert info.solve_time_seconds >= 0
        assert info.peak_memory_bytes is not None
        assert info.peak_memory_bytes >= 0
    _assert_under(results.output_dir, tmp_path)
    for plot_path in results.plot_paths.to_mapping().values():
        _assert_under(plot_path, tmp_path)
