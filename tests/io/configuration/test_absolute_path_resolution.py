"""Absolute paths survive config path resolution on a real directory."""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.io import url_resolver

from neuralls.platform.config.context import ConfigContext, expand_config_path
from neuralls.platform.config.models.data_models import SourceConfig
from neuralls.platform.config.resolution import resolve_local_path


def test_absolute_path_unchanged(config_context: ConfigContext, tmp_path: Path) -> None:
    """Absolute paths pass through unchanged after resolution."""
    absolute = tmp_path / "matrix.mtx"
    value = expand_config_path(str(absolute), config_context)
    assert value == str(absolute.resolve())


def test_source_config_matrix_path_plain_still_resolves(
    config_context: ConfigContext, tmp_path: Path
) -> None:
    """matrix_path without wildcards resolves to an absolute path as before."""
    plain = str((tmp_path / "workspace") / "A.txt")
    config = SourceConfig.model_validate(
        {"matrix_path": plain},
        context=config_context.as_pydantic_context(),
    )
    assert config.matrix_path is not None
    assert "*" not in config.matrix_path
    assert config.matrix_path == plain


def test_resolve_local_path_accepts_file_uri(tmp_path: Path) -> None:
    """file:// URIs round-trip through the shared local resolver."""
    expected = (tmp_path / "mlartifacts/run-1/artifacts").resolve()
    uri = url_resolver.build_uri(expected, scheme="file")

    resolved = resolve_local_path(uri, base_dir=Path.cwd())

    assert resolved == expected
