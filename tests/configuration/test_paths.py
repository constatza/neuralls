"""Tests for path context construction."""

from __future__ import annotations

from pathlib import Path

from neuralls.platform.config.models.data_models import DataConfigFile, OutputConfig
from neuralls.platform.config.resolution import (
    resolve_path_context,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT

WORKSPACE = Path("workspace")


def test_resolve_path_context_uses_settings_defaults(
    neuralls_settings: NeurallsSettings,
) -> None:
    """Default roots come from NeurallsSettings."""
    data_cfg = DataConfigFile(id="test", output=OutputConfig())
    ctx = resolve_path_context(
        processed_root=neuralls_settings.processed_dir,
        output_root=neuralls_settings.output_dir,
        data_dir_override=data_cfg.output.data_dir,
    )
    assert ctx.project_root == DEFAULT_PROJECT_ROOT
    assert ctx.output_root == neuralls_settings.output_dir
    assert ctx.processed_root == neuralls_settings.processed_dir


def test_resolve_path_context_honours_output_override(
    neuralls_settings: NeurallsSettings,
) -> None:
    """Explicit output overrides beat settings.output_dir."""
    custom_output = WORKSPACE / "custom-output"
    data_cfg = DataConfigFile(id="test", output=OutputConfig())
    ctx = resolve_path_context(
        processed_root=neuralls_settings.processed_dir,
        output_root=neuralls_settings.output_dir,
        data_dir_override=data_cfg.output.data_dir,
        output_override=custom_output,
    )
    assert ctx.output_root == custom_output


def test_resolve_path_context_honours_data_dir_override(
    neuralls_settings: NeurallsSettings,
) -> None:
    """Dataset-level data_dir overrides settings.processed_dir."""
    processed_dir = WORKSPACE / "processed"
    data_cfg = DataConfigFile(id="test", output=OutputConfig(data_dir=processed_dir))
    ctx = resolve_path_context(
        processed_root=neuralls_settings.processed_dir,
        output_root=neuralls_settings.output_dir,
        data_dir_override=data_cfg.output.data_dir,
    )
    assert ctx.processed_root == processed_dir
