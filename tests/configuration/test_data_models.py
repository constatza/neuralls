"""Tests for typed dataset configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.models.data_models import (
    GenerationConfig,
    SourceConfig,
)
from tests.workflows.comparison_support import VIRTUAL_CASE_PATH, VIRTUAL_HOME, virtual_settings


def test_source_config_matrix_path_glob_preserves_wildcard() -> None:
    """matrix_path with a glob wildcard is anchored to the config directory, wildcard intact."""
    context = ConfigContext(
        config_path=VIRTUAL_CASE_PATH, settings=virtual_settings(), home_dir=VIRTUAL_HOME
    )
    glob_expr = str(Path("data") / "A_*.txt")
    config = SourceConfig.model_validate(
        {"matrix_path": glob_expr}, context=context.as_pydantic_context()
    )
    assert config.matrix_path == str(VIRTUAL_CASE_PATH.parent / glob_expr)


@pytest.mark.parametrize("value", [0, -1])
def test_generation_write_batch_size_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValidationError):
        GenerationConfig(write_batch_size=value)
