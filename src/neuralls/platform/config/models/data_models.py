"""Pydantic models for data configuration validation.

This module provides Pydantic models for validating data config TOML structure.
These models validate the structure of data generation/collection configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from neuralls.domain.generation.source_streams import EnumerateBy
from neuralls.platform.config.context import ConfigContext, expand_config_glob, expand_config_path
from neuralls.shared.digest import Cosmetic
from neuralls.shared.types import DatasetFormat


class SourceConfig(BaseModel):
    """Validates [source] section from data config.

    Specifies source data locations for matrix and solution vectors.
    """

    type: Literal["rhs_archive", "generated", "solution_archive"] | None = Field(
        default=None,
        description="Source data type",
    )
    case_path: Annotated[str | None, Cosmetic()] = Field(
        default=None,
        description="Path to case directory",
    )
    matrix_file: str | None = Field(
        default=None,
        description="Matrix filename (relative to case_path)",
    )
    matrix_path: Annotated[str | None, Cosmetic()] = Field(
        default=None,
        description="Full path to matrix file",
    )
    rhs_path: Annotated[str | None, Cosmetic()] = Field(
        default=None,
        description="Path to RHS vector files",
    )
    rhs_pattern: str | None = Field(
        default=None,
        description="Glob pattern for RHS files",
    )
    solutions_path: Annotated[str | None, Cosmetic()] = Field(
        default=None,
        description="Path or glob pattern for solution vectors",
    )
    solution_path: Annotated[str | None, Cosmetic()] = Field(
        default=None,
        description=(
            "Path or glob pattern for per-matrix solution vectors used for solution binding. "
            "Each file is paired with its matrix by the same sample_id_regex/enumerate_by strategy. "
            "Distinct from solutions_path (which feeds the solution_archive strategy glob)."
        ),
    )
    sample_id_regex: str | None = Field(
        default=None,
        description="Regex used to extract sample IDs from glob filenames for source pairing",
    )
    enumerate_by: EnumerateBy | None = Field(
        default=None,
        description=(
            "Sort glob-matched files by this criterion and assign sequential IDs (0, 1, 2, …). "
            "Use when filenames carry no natural integer ID. "
            "Mutually exclusive with sample_id_regex."
        ),
    )
    parameters_paths: Annotated[tuple[str, ...], Cosmetic()] = Field(
        default=(),
        max_length=2,
        description=(
            "Glob patterns for per-matrix parameters files (max 2). "
            "Each path is paired with its matrix by the same sample_id_regex/enumerate_by strategy. "
            "Written to the dataset as parameters_0.zarr, parameters_1.zarr, …"
        ),
    )
    include_indices: tuple[int, ...] | None = Field(
        default=None,
        description=(
            "Restrict glob-matched matrix/parameters samples to exactly these sample IDs "
            "(as assigned by sample_id_regex/enumerate_by). Mutually exclusive with "
            "exclude_indices. Use to build a comparison/holdout dataset from a subset of a "
            "parametric matrix family."
        ),
    )
    exclude_indices: tuple[int, ...] = Field(
        default=(),
        description=(
            "Drop these sample IDs from glob-matched matrix/parameters samples before "
            "generation. Use to keep a parametric matrix family's held-out members out of "
            "training/POD-snapshot datasets. Mutually exclusive with include_indices."
        ),
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _check_id_strategy_exclusivity(self) -> SourceConfig:
        """Raise when both sample_id_regex and enumerate_by are set."""
        if self.sample_id_regex is not None and self.enumerate_by is not None:
            raise ValueError(
                "sample_id_regex and enumerate_by are mutually exclusive: "
                "use one or the other, not both."
            )
        return self

    @model_validator(mode="after")
    def _check_index_filter_exclusivity(self) -> SourceConfig:
        """Raise when both include_indices and exclude_indices are set."""
        if self.include_indices is not None and self.exclude_indices:
            raise ValueError(
                "include_indices and exclude_indices are mutually exclusive: "
                "use one or the other, not both."
            )
        return self

    @field_validator("case_path", mode="before")
    @classmethod
    def _expand_plain_paths(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Expand plain path placeholders and resolve to absolute paths.

        Args:
            v: Raw string value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or None if v is None.
        """
        if v is None or info.context is None:
            return v
        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @field_validator(
        "rhs_path", "solutions_path", "rhs_pattern", "matrix_path", "solution_path", mode="before"
    )
    @classmethod
    def _expand_glob_paths(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Expand glob path placeholders, preserving wildcard characters.

        Also handles plain paths without wildcards via fallback to expand_config_path.

        Args:
            v: Raw string value or glob expression from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved path or glob expression, or None if v is None.
        """
        if v is None or info.context is None:
            return v
        return expand_config_glob(v, ConfigContext.from_pydantic_context(info.context))

    @field_validator("parameters_paths", mode="before")
    @classmethod
    def _expand_parameters_paths(cls, v: object, info: ValidationInfo) -> tuple[str, ...]:
        """Expand glob placeholders in each parameters path, preserving wildcards.

        Args:
            v: Raw sequence of path strings from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Tuple of resolved glob expressions.
        """
        items: tuple[str, ...] = tuple(str(p) for p in v) if isinstance(v, (list, tuple)) else ()
        if not items or info.context is None:
            return items
        ctx = ConfigContext.from_pydantic_context(info.context)
        return tuple(expand_config_glob(p, ctx) for p in items)


class StrategyConfig(BaseModel):
    """Validates [[generation.strategy]] entry from data config.

    Each strategy defines a data generation approach (e.g., residual_traces, solution_archive).
    """

    name: str = Field(
        ...,
        description="Strategy name (residual_traces, residuals, gaussian_residuals, solution_archive, etc.)",
    )
    samples: int = Field(
        default=0,
        description="Number of samples to generate (-1 for all available)",
    )
    krylov_iters: int | None = Field(
        default=None,
        description="Number of Krylov iterations for krylov-based strategies",
        ge=1,
    )
    residual_iters: int | None = Field(
        default=None,
        description="Number of residual iterations for residual-based strategies",
        ge=1,
    )
    solutions_glob: Annotated[str | None, Cosmetic()] = Field(
        default=None,
        description="Glob pattern for solution vector files",
    )
    rhs_glob: Annotated[str | None, Cosmetic()] = Field(
        default=None,
        description="Glob pattern for RHS vector files",
    )

    model_config = ConfigDict(extra="allow", frozen=True)  # Allow strategy-specific parameters

    @field_validator("solutions_glob", "rhs_glob", mode="before")
    @classmethod
    def _expand_globs(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Expand glob path placeholders, preserving wildcard characters.

        Args:
            v: Raw glob expression from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Glob expression with prefix resolved to absolute path, or None if v is None.
        """
        if v is None or info.context is None:
            return v
        return expand_config_glob(v, ConfigContext.from_pydantic_context(info.context))

    @model_validator(mode="before")
    @classmethod
    def _expand_extra_archive_paths(cls, data: object, info: ValidationInfo) -> object:
        """Expand extra rhs_path / solutions_path keys when strategy extras use them."""
        if not isinstance(data, dict) or info.context is None:
            return data
        expanded = dict(data)
        ctx = ConfigContext.from_pydantic_context(info.context)
        for key in ("rhs_path", "solutions_path"):
            value = expanded.get(key)
            if isinstance(value, str):
                expanded[key] = expand_config_glob(value, ctx)
        return expanded


class GenerationConfig(BaseModel):
    """Validates [generation] section from data config.

    Controls data generation/collection behavior and normalization.
    """

    normalize: str | bool = Field(
        default="matrix",
        description="Normalization strategy (matrix, rhs, or none)",
    )
    shuffle: bool = Field(
        default=True,
        description="Shuffle generated samples",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    replacement: bool = Field(
        default=False,
        description=(
            "Whether global multi-matrix generation may reuse matrix bindings for "
            "supported random strategies."
        ),
    )
    num_samples: int | None = Field(
        default=None,
        description="Total number of samples (legacy, use strategy.samples)",
        ge=1,
    )
    rhs_archive_glob: Annotated[str | None, Cosmetic()] = Field(
        default=None,
        description="Optional generation-level RHS archive glob",
    )
    cg_tolerance: float | None = Field(
        default=None,
        description="Optional generation-level CG tolerance override",
        gt=0.0,
    )
    cg_max_iters: int | None = Field(
        default=None,
        description="Optional generation-level CG iteration override",
        ge=1,
    )
    strategy: list[StrategyConfig] = Field(
        default_factory=list,
        description="List of generation strategies",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("rhs_archive_glob", mode="before")
    @classmethod
    def _expand_rhs_archive_glob(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Expand generation-level RHS archive globs."""
        if v is None or info.context is None:
            return v
        return expand_config_glob(v, ConfigContext.from_pydantic_context(info.context))


class OutputConfig(BaseModel):
    """Validates [output] section from data config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Annotated[Path | None, Cosmetic()] = Field(
        default=None,
        description="Output directory for generated data",
    )
    dataset_format: DatasetFormat = Field(
        default="hdf5",
        description="Dataset storage format ('zarr', 'npy', or 'hdf5')",
    )
    matrix_codec: Literal["coo"] = Field(
        default="coo",
        description="Sparse matrix codec",
    )
    matrix_replication: Literal["duplicate_per_sample"] = Field(
        default="duplicate_per_sample",
        description="Matrix replication policy",
    )
    dtype: Literal["float64"] = Field(
        default="float64",
        description="Numeric dtype for persisted arrays",
    )

    def with_data_dir(self, data_dir: Path) -> OutputConfig:
        """Return a copy of this config with data_dir replaced.

        Args:
            data_dir: Absolute path to use as the new output directory.

        Returns:
            A new frozen OutputConfig with only data_dir replaced.
        """
        return self.model_copy(update={"data_dir": data_dir})

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_data_dir(cls, v: object, info: ValidationInfo) -> object:
        """Expand data_dir placeholder and resolve to absolute path.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if v is None or info.context is None:
            return v
        if not isinstance(v, str):
            return v
        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))


class DataTestConfig(BaseModel):
    """Validates [test] section from data config.

    Optional section specifying test data locations.
    """

    solutions_path: str | None = Field(
        default=None,
        description="Path or glob pattern for test solution vectors",
    )
    rhs_path: str | None = Field(
        default=None,
        description="Path or glob pattern for test RHS vectors",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("solutions_path", "rhs_path", mode="before")
    @classmethod
    def _expand_glob_paths(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Expand glob path placeholders, preserving wildcard characters.

        Args:
            v: Raw glob expression from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Glob expression with prefix resolved to absolute path, or None if v is None.
        """
        if v is None or info.context is None:
            return v
        return expand_config_glob(v, ConfigContext.from_pydantic_context(info.context))


class DataConfigFile(BaseModel):
    """Complete data config TOML structure with validation.

    This model validates the top-level sections of a data configuration file
    used for data generation and collection workflows.
    """

    id: Annotated[str, Cosmetic()] = Field(
        ..., min_length=1, description="Stable dataset identifier (label, not identity)"
    )
    source: SourceConfig = Field(
        default_factory=SourceConfig,
        description="Source data locations",
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Data generation configuration",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output paths configuration",
    )
    test: DataTestConfig = Field(
        default_factory=DataTestConfig,
        description="Test data configuration",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)
