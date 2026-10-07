"""Tests for typed dataset configuration models."""

from __future__ import annotations

from pathlib import Path, PurePath

import pytest
from pydantic import ValidationError

from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.models.data_models import (
    DataConfigFile,
    DataTestConfig,
    GenerationConfig,
    OutputConfig,
    SourceConfig,
    StrategyConfig,
)
from neuralls.shared.constants import DEFAULT_WRITE_BATCH_SIZE


def test_source_config_matrix_path_glob_preserves_wildcard(
    config_context: ConfigContext,
    tmp_path: Path,
) -> None:
    """matrix_path with a glob wildcard must survive validator expansion intact."""
    glob_expr = str(tmp_path / "A_*.txt")
    config = SourceConfig.model_validate(
        {"matrix_path": glob_expr},
        context=config_context.as_pydantic_context(),
    )
    assert config.matrix_path is not None
    assert "*" in config.matrix_path
    assert PurePath(config.matrix_path).name == "A_*.txt"


def test_source_config_matrix_path_plain_still_resolves(
    config_context: ConfigContext,
    tmp_path: Path,
) -> None:
    """matrix_path without wildcards resolves to an absolute path as before."""
    plain = str(tmp_path / "A.txt")
    config = SourceConfig.model_validate(
        {"matrix_path": plain},
        context=config_context.as_pydantic_context(),
    )
    assert config.matrix_path is not None
    assert "*" not in config.matrix_path
    assert config.matrix_path == plain


def test_source_config_defaults() -> None:
    """Source config defaults to optional paths."""
    config = SourceConfig()
    assert config.type is None
    assert config.matrix_path is None
    assert config.solutions_path is None


def test_strategy_config_keeps_required_name() -> None:
    """Strategy config requires only a name and defaults samples to zero."""
    config = StrategyConfig(name="residual_traces")
    assert config.name == "residual_traces"
    assert config.samples == 0


def test_generation_config_defaults() -> None:
    """Generation config exposes stable defaults."""
    config = GenerationConfig()
    assert config.normalize == "matrix"
    assert config.shuffle is True
    assert config.seed == 42
    assert config.strategy == []


def test_generation_write_batch_size_defaults_to_shared_constant() -> None:
    """Batch size defaults to the shared constant so unconfigured runs share one bound."""
    assert GenerationConfig().write_batch_size == DEFAULT_WRITE_BATCH_SIZE


@pytest.mark.parametrize("value", [0, -1])
def test_generation_write_batch_size_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValidationError, match="write_batch_size"):
        GenerationConfig(write_batch_size=value)


def test_generation_write_batch_size_accepts_positive_override() -> None:
    assert GenerationConfig(write_batch_size=7).write_batch_size == 7


def test_data_test_config_defaults() -> None:
    """Optional test metadata defaults to empty paths."""
    config = DataTestConfig()
    assert config.solutions_path is None
    assert config.rhs_path is None


def test_data_config_file_requires_id() -> None:
    """Dataset configs require a stable id."""
    with pytest.raises(ValidationError, match="id"):
        DataConfigFile()  # ty: ignore[missing-argument]


def test_data_config_file_round_trip() -> None:
    """Typed dataset config retains all structured sections."""
    config = DataConfigFile(
        id="test-dataset",
        source=SourceConfig(
            type="solution_archive",
            matrix_path="tests/fixtures/data/matrix.txt",
            solutions_path="tests/fixtures/data/solutions/*.txt",
        ),
        generation=GenerationConfig(
            normalize="matrix",
            shuffle=True,
            seed=42,
            strategy=[StrategyConfig(name="solution_archive", samples=-1, solutions_glob="*.txt")],
        ),
        output=OutputConfig(data_dir=Path("tests/fixtures/data/processed")),
        test=DataTestConfig(solutions_path="tests/fixtures/data/test_solutions.txt"),
    )
    assert config.id == "test-dataset"
    assert config.source.type == "solution_archive"
    assert config.generation.strategy[0].name == "solution_archive"
    assert config.output.data_dir == Path("tests/fixtures/data/processed")
    assert config.test.solutions_path == "tests/fixtures/data/test_solutions.txt"


def test_data_config_file_is_frozen() -> None:
    """Typed dataset configs remain immutable."""
    config = DataConfigFile(id="frozen")
    with pytest.raises((ValidationError, AttributeError)):
        config.id = "mutated"


def test_data_config_file_forbids_extra_fields() -> None:
    """Unknown top-level fields are rejected."""
    with pytest.raises(ValidationError):
        DataConfigFile(id="test", custom_field="custom_value")
