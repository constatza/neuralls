"""Tests for typed dataset configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.models.data_models import (
    DataTestConfig,
    GenerationConfig,
    SourceConfig,
    StrategyConfig,
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


_RAW_GLOB = "${NEURALLS_RAW_DIR}/b_*.txt"


@pytest.mark.parametrize(
    ("model", "payload", "field"),
    [
        pytest.param(
            StrategyConfig,
            {"name": "rhs_archive", "rhs_glob": _RAW_GLOB},
            "rhs_glob",
            id="strategy rhs glob",
        ),
        pytest.param(DataTestConfig, {"rhs_path": _RAW_GLOB}, "rhs_path", id="test data rhs path"),
    ],
)
def test_a_glob_validated_without_a_config_context_is_kept_as_written(
    model: type[StrategyConfig | DataTestConfig], payload: dict[str, str], field: str
) -> None:
    """With no context to expand placeholders from, the glob passes through untouched."""
    built = model.model_validate(payload)

    assert getattr(built, field) == _RAW_GLOB
