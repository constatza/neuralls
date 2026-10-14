"""Regression tests for test filesystem scope policy."""

from __future__ import annotations

from pathlib import Path

from neuralls.platform.config.loaders import load_comparison_config


def test_runtime_roots_stay_under_tmp_path(
    tmp_path: Path,
    runtime_root: Path,
    output_root: Path,
    processed_root: Path,
    figures_root: Path,
    mlflow_tracking_dir: Path,
    mlflow_artifact_dir: Path,
) -> None:
    """All canonical runtime paths should be sandboxed under the test temp dir."""
    expected_root = (tmp_path / "runtime").resolve()

    assert runtime_root == expected_root
    assert output_root == expected_root / "output"
    assert processed_root == expected_root / "processed"
    assert figures_root == expected_root / "figures"
    assert mlflow_tracking_dir == expected_root / "mlruns"
    assert mlflow_artifact_dir == expected_root / "mlartifacts"
    assert Path.cwd() == tmp_path.resolve()


def test_comparison_fixture_configs_load_from_tests_tree(neuralls_settings) -> None:
    """Comparison loader coverage should use test-owned fixture configs."""
    comparison_dir = Path(__file__).resolve().parent / "fixtures" / "configs" / "comparison"

    linear = load_comparison_config(comparison_dir / "linear.toml", neuralls_settings)
    ffnn = load_comparison_config(comparison_dir / "ffnn.toml", neuralls_settings)

    assert len(linear.preconditioners) == 1
    assert len(ffnn.preconditioners) == 1
