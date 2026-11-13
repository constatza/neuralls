"""Absolute paths survive config path resolution on a real directory."""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.io import url_resolver

from neuralls.platform.config.context import ConfigContext, expand_config_path
from neuralls.platform.config.models.data_models import SourceConfig
from neuralls.platform.config.resolution import resolve_local_path


def test_absolute_path_unchanged(nested_config_context: ConfigContext, tmp_path: Path) -> None:
    """An absolute path outside the config directory is not re-anchored to it."""
    absolute = tmp_path / "elsewhere" / "matrix.mtx"
    value = expand_config_path(str(absolute), nested_config_context)
    assert value == str(absolute.resolve())


def test_source_config_matrix_path_plain_still_resolves(
    nested_config_context: ConfigContext,
) -> None:
    """A relative matrix_path without wildcards is anchored to the config directory."""
    config = SourceConfig.model_validate(
        {"matrix_path": "workspace/A.txt"},
        context=nested_config_context.as_pydantic_context(),
    )
    assert config.matrix_path == str(
        (nested_config_context.config_dir / "workspace/A.txt").resolve()
    )


def test_resolve_local_path_accepts_file_uri(tmp_path: Path) -> None:
    """file:// URIs round-trip through the shared local resolver."""
    expected = (tmp_path / "mlartifacts/run-1/artifacts").resolve()
    uri = url_resolver.build_uri(expected, scheme="file")

    resolved = resolve_local_path(uri, base_dir=tmp_path / "unrelated")

    assert resolved == expected
