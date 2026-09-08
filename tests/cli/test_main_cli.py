"""Tests for the public neuralls CLI surface."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast, get_args, get_type_hints
from unittest.mock import MagicMock, patch

from typer.core import TyperGroup as Group
from typer.main import get_command
from typer.models import ArgumentInfo
from typer.testing import CliRunner

from neuralls.application.models import AssignmentResult
from neuralls.cli.compare import compare_case
from neuralls.cli.eval import eval_case_batch
from neuralls.cli.generate import generate_case
from neuralls.cli.generate_single import generate_single
from neuralls.cli.main import app
from neuralls.cli.run import run_case_pipeline_command
from neuralls.cli.train import train_case_batch
from neuralls.composition.comparison.models import (
    ComparisonOutcome,
    ComparisonParams,
    ComparisonResult,
)
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.result import ComparisonRecommendations
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.shared.constants import EXIT_FAILURE

runner = CliRunner()


def _solver_params(tmp_path: Path) -> ComparisonGeneral:
    return ComparisonGeneral(
        params=SolverParams(
            rtol=1.0e-6,
            atol=1.0e-14,
            max_iterations=10,
            stopping_criterion="residual_norm",
            m_max=20,
        ),
        data=ComparisonData(
            matrix_path=tmp_path / "matrix.npy",
            rhs_path=tmp_path / "rhs.npy",
        ),
    )


def _comparison_payload(tmp_path: Path) -> ComparisonResult:
    return ComparisonResult(
        results={},
        summary="ok",
        solver_params=_solver_params(tmp_path),
        preconditioners=("none",),
        recommendations=ComparisonRecommendations(),
    )


def test_root_help_lists_only_public_commands() -> None:
    command = get_command(app)
    assert isinstance(command, Group)

    assert sorted(command.commands) == [
        "compare",
        "config",
        "eval",
        "generate",
        "generate-single",
        "run",
        "train",
    ]


def test_config_subcommands_are_exposed_under_root() -> None:
    root_command = get_command(app)
    assert isinstance(root_command, Group)

    assert "config" in root_command.commands


def test_project_scripts_only_expose_neuralls() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"] == {"neuralls": "neuralls.cli.main:app"}


def test_generate_help_shows_batch_mode() -> None:
    command = get_command(app)
    assert isinstance(command, Group)
    parameter = cast(Any, command.commands["generate"].params[0])

    assert parameter.help == "Path to a case config TOML."


def test_generate_single_help_shows_dataset_mode() -> None:
    command = get_command(app)
    assert isinstance(command, Group)
    parameter = cast(Any, command.commands["generate-single"].params[0])

    assert parameter.help == "Path to a dataset config TOML."


def test_generate_signature_uses_batch_case_argument() -> None:
    config = get_args(get_type_hints(generate_case, include_extras=True)["config"])[1]

    assert isinstance(config, ArgumentInfo)
    assert config.default is ...


@patch("neuralls.cli.generate.generate_batch")
@patch("neuralls.cli.generate.load_validated_case_config")
@patch("neuralls.cli.generate.load_case_settings")
def test_generate_invokes_batch_workflow(
    mock_load_settings: MagicMock,
    mock_load_case_config: MagicMock,
    mock_generate_batch: MagicMock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "case.toml"
    config.write_text("", encoding="utf-8")
    settings = MagicMock()
    cfg = MagicMock()
    mock_load_settings.return_value = settings
    mock_load_case_config.return_value = (cfg, MagicMock())
    mock_generate_batch.return_value = [
        MagicMock(dataset_id="residuals", output_dir=tmp_path / "out")
    ]

    result = runner.invoke(app, ["generate", str(config)])

    assert result.exit_code == 0
    mock_load_settings.assert_called_once_with(config, None, profile=None)
    mock_load_case_config.assert_called_once_with(config, settings)
    mock_generate_batch.assert_called_once_with(
        cfg=cfg,
        configs_dir=config.resolve().parent,
        settings=settings,
        force=False,
    )


@patch("neuralls.cli.generate.generate_batch")
@patch("neuralls.cli.generate.load_validated_case_config")
@patch("neuralls.cli.generate.load_case_settings")
def test_generate_fails_for_case_config_without_datasets(
    mock_load_settings: MagicMock,
    mock_load_case_config: MagicMock,
    mock_generate_batch: MagicMock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "case.toml"
    config.write_text("", encoding="utf-8")
    settings = MagicMock()
    cfg = MagicMock(datasets=[])
    mock_load_settings.return_value = settings
    mock_load_case_config.return_value = (cfg, MagicMock())

    result = runner.invoke(app, ["generate", str(config)])

    assert result.exit_code == EXIT_FAILURE
    mock_load_settings.assert_called_once_with(config, None, profile=None)
    mock_load_case_config.assert_called_once_with(config, settings)
    mock_generate_batch.assert_not_called()
    assert "Error during batch generation [ValueError]:" in result.stderr


@patch("neuralls.cli.generate.generate_batch")
@patch("neuralls.cli.generate.load_validated_case_config")
@patch("neuralls.cli.generate.load_case_settings")
def test_generate_fails_for_dataset_config_without_single_subcommand(
    mock_load_settings: MagicMock,
    mock_load_case_config: MagicMock,
    mock_generate_batch: MagicMock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "dataset.toml"
    config.write_text(
        "\n".join(
            [
                "[source]",
                f'matrix_path = "{(tmp_path / "matrix.txt").as_posix()}"',
                "",
                "[generation]",
                "shuffle = true",
                "",
                "[output]",
                f'data_dir = "{(tmp_path / "processed").as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["generate", str(config)])

    assert result.exit_code == EXIT_FAILURE
    mock_load_settings.assert_not_called()
    mock_load_case_config.assert_not_called()
    mock_generate_batch.assert_not_called()
    assert "Error during batch generation [ValueError]:" in result.stderr


def test_generate_single_signature_uses_dataset_argument_and_case_option() -> None:
    config = get_args(get_type_hints(generate_single, include_extras=True)["config"])[1]

    assert isinstance(config, ArgumentInfo)
    assert config.default is ...


@patch("neuralls.cli.generate_single.process_data_from_config")
@patch("neuralls.cli.generate_single.load_case_settings")
def test_generate_single_invokes_single_dataset_workflow(
    mock_load_settings: MagicMock,
    mock_process_data: MagicMock,
    tmp_path: Path,
) -> None:
    dataset_config = tmp_path / "dataset.toml"
    case_config = tmp_path / "case.toml"
    output_dir = tmp_path / "processed" / "dataset"
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    dataset_config.write_text("", encoding="utf-8")
    case_config.write_text("", encoding="utf-8")
    settings = MagicMock()
    mock_load_settings.return_value = settings
    mock_process_data.return_value = output_dir

    result = runner.invoke(
        app,
        ["generate-single", str(dataset_config), "--case-config", str(case_config)],
    )

    assert result.exit_code == 0
    mock_load_settings.assert_called_once_with(case_config, None, profile=None)
    mock_process_data.assert_called_once_with(dataset_config, settings, force=False)


@patch("neuralls.cli.generate_single.process_data_from_config")
@patch("neuralls.cli.generate_single.load_case_settings")
def test_generate_single_requires_case_config(
    mock_load_settings: MagicMock,
    mock_process_data: MagicMock,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_config = tmp_path / "dataset.toml"
    dataset_config.write_text("", encoding="utf-8")
    monkeypatch.delenv("NEURALLS_CASE_CONFIG", raising=False)

    result = runner.invoke(app, ["generate-single", str(dataset_config)])

    assert result.exit_code == EXIT_FAILURE
    mock_load_settings.assert_not_called()
    mock_process_data.assert_not_called()
    assert "Error during data generation [ValueError]:" in result.stderr


@patch("neuralls.cli.generate.generate_batch")
@patch("neuralls.cli.generate.load_validated_case_config")
@patch("neuralls.cli.generate.load_case_settings")
def test_generate_preserves_enriched_storage_error_message(
    mock_load_settings: MagicMock,
    mock_load_case_config: MagicMock,
    mock_generate_batch: MagicMock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "case.toml"
    config.write_text("", encoding="utf-8")
    settings = MagicMock()
    cfg = MagicMock(datasets=[MagicMock()])
    mock_load_settings.return_value = settings
    mock_load_case_config.return_value = (cfg, MagicMock())
    matrix_store = tmp_path / "matrix.zarr"
    mock_generate_batch.side_effect = OSError(
        f"Updating matrix.zarr store at {matrix_store} failed. "
        "PermissionError: denied, winerror=5, src=a.partial, dst=zarr.json"
    )

    result = runner.invoke(app, ["generate", str(config)])

    assert result.exit_code == EXIT_FAILURE
    assert "Error during batch generation [OSError]:" in result.stderr
    assert "winerror=5" in result.stderr
    assert "dst=zarr.json" in result.stderr


@patch("neuralls.cli.generate_single.process_data_from_config")
@patch("neuralls.cli.generate_single.load_case_settings")
def test_generate_single_preserves_enriched_storage_error_message(
    mock_load_settings: MagicMock,
    mock_process_data: MagicMock,
    tmp_path: Path,
) -> None:
    dataset_config = tmp_path / "dataset.toml"
    case_config = tmp_path / "case.toml"
    dataset_config.write_text("", encoding="utf-8")
    case_config.write_text("", encoding="utf-8")
    settings = MagicMock()
    mock_load_settings.return_value = settings
    rhs_store = tmp_path / "rhs.zarr"
    mock_process_data.side_effect = PermissionError(
        f"Writing rhs.zarr at {rhs_store} failed. "
        "PermissionError: denied, winerror=5, src=b.partial, dst=zarr.json"
    )

    result = runner.invoke(
        app,
        ["generate-single", str(dataset_config), "--case-config", str(case_config)],
    )

    assert result.exit_code == EXIT_FAILURE
    assert "Error during data generation [PermissionError]:" in result.stderr
    assert "winerror=5" in result.stderr


def test_train_signature_uses_batch_case_argument() -> None:
    config = get_args(get_type_hints(train_case_batch, include_extras=True)["config"])[1]

    assert isinstance(config, ArgumentInfo)
    assert config.default is ...


def test_eval_signature_uses_batch_case_argument() -> None:
    config = get_args(get_type_hints(eval_case_batch, include_extras=True)["config"])[1]

    assert isinstance(config, ArgumentInfo)
    assert config.default is ...


@patch("neuralls.cli.train.write_metric_report")
@patch("neuralls.cli.train.run_assignment_sweep")
@patch("neuralls.cli.train.load_case_settings")
def test_train_invokes_batch_workflow(
    mock_load_settings: MagicMock,
    mock_run_assignment_sweep: MagicMock,
    mock_write_metric_report: MagicMock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "case.toml"
    config.write_text("", encoding="utf-8")
    settings = MagicMock()
    sweep_result = MagicMock(results=[], parent_run_id="parent-run-1")
    mock_load_settings.return_value = settings
    mock_run_assignment_sweep.return_value = sweep_result
    mock_write_metric_report.return_value = True

    result = runner.invoke(app, ["train", str(config)])

    assert result.exit_code == 0
    mock_load_settings.assert_called_once_with(config, None, profile=None)
    mock_run_assignment_sweep.assert_called_once_with(
        case_config_path=config.resolve(),
        settings=settings,
        force=False,
    )
    mock_write_metric_report.assert_called_once_with(sweep_result, metric="eval/mae")


@patch("neuralls.cli.eval.write_eval_metric_report")
@patch("neuralls.cli.eval.eval_batch")
@patch("neuralls.cli.eval.load_validated_case_config")
@patch("neuralls.cli.eval.load_case_settings")
def test_eval_invokes_batch_workflow(
    mock_load_settings: MagicMock,
    mock_load_case_config: MagicMock,
    mock_eval_batch: MagicMock,
    mock_write_metric_report: MagicMock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "case.toml"
    config.write_text("", encoding="utf-8")
    settings = MagicMock()
    cfg = MagicMock()
    batch = MagicMock()
    mock_load_settings.return_value = settings
    mock_load_case_config.return_value = (cfg, MagicMock())
    mock_eval_batch.return_value = batch
    mock_write_metric_report.return_value = True

    result = runner.invoke(
        app,
        ["eval", str(config), "--assignment", "a1", "--assignment", "a2"],
    )

    assert result.exit_code == 0
    mock_load_settings.assert_called_once_with(config, None, profile=None)
    mock_load_case_config.assert_called_once_with(config, settings)
    mock_eval_batch.assert_called_once_with(
        cfg=cfg,
        configs_dir=config.resolve().parent,
        settings=settings,
        output_root=None,
        case_config_path=config.resolve(),
        assignment_ids=["a1", "a2"],
    )
    mock_write_metric_report.assert_called_once_with(batch, metric="mae")


def test_run_signature_uses_batch_case_argument() -> None:
    config = get_args(get_type_hints(run_case_pipeline_command, include_extras=True)["config"])[1]

    assert isinstance(config, ArgumentInfo)
    assert config.default is ...


@patch("neuralls.cli.run.run_case_pipeline")
@patch("neuralls.cli.run.load_case_settings")
def test_run_invokes_batch_workflow(
    mock_load_settings: MagicMock,
    mock_run_case_pipeline: MagicMock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "case.toml"
    config.write_text("", encoding="utf-8")
    settings = MagicMock()
    mock_load_settings.return_value = settings
    mock_run_case_pipeline.return_value = (
        [
            AssignmentResult(
                assignment_id="exp-1", assignment_display_name="exp-1", status="Success"
            )
        ],
        [],
    )

    result = runner.invoke(app, ["run", str(config)])

    assert result.exit_code == 0
    mock_load_settings.assert_called_once_with(config, None, profile=None)
    mock_run_case_pipeline.assert_called_once()
    call_kwargs = mock_run_case_pipeline.call_args.kwargs
    assert call_kwargs["case_config_path"] == config
    assert call_kwargs["settings"] == settings
    assert call_kwargs["force_train"] is False
    assert call_kwargs["force_generate"] is False
    assert call_kwargs["force_compare"] is False
    assert call_kwargs["max_epochs"] is None


def test_compare_signature_uses_batch_case_argument() -> None:
    config = get_args(get_type_hints(compare_case, include_extras=True)["config"])[1]

    assert isinstance(config, ArgumentInfo)
    assert config.default is ...


@patch("neuralls.cli.compare.run_comparison_batch")
@patch("neuralls.cli.compare.load_case_settings")
def test_compare_invokes_batch_workflow(
    mock_load_settings: MagicMock,
    mock_run_comparison_batch: MagicMock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "case.toml"
    config.write_text(
        "[mlflow]\n"
        f'tracking_uri = "{build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")}"\n',
        encoding="utf-8",
    )
    settings = MagicMock()
    mock_load_settings.return_value = settings
    mock_run_comparison_batch.return_value = [
        ComparisonOutcome(
            comparison_id="solver",
            comparison_display_name="solver",
            success=True,
            payload=_comparison_payload(tmp_path),
        )
    ]

    result = runner.invoke(app, ["compare", str(config)])

    assert result.exit_code == 0
    mock_load_settings.assert_called_once_with(config, None, profile=None)
    mock_run_comparison_batch.assert_called_once()
    assert mock_run_comparison_batch.call_args.args[0] == config
    assert isinstance(mock_run_comparison_batch.call_args.args[1], ComparisonParams)
    assert mock_run_comparison_batch.call_args.args[2] == settings
