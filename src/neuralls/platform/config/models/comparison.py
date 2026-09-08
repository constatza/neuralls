"""Comparison configuration dataclasses and strict parser models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from neuralls.domain.solver.models.config import (
    ComparisonData,
    ComparisonGeneral,
    SolverParams,
)
from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    RegisteredModelRefConfig,
)
from neuralls.shared.constants import (
    DEFAULT_ATOL,
    DEFAULT_M_MAX,
    DEFAULT_RTOL,
)
from neuralls.shared.types import RowKind

type NormalizeSystem = Literal["none", "matrix", "rhs", "both"]


@dataclass(frozen=True)
class ComparisonConfig:
    """Runtime domain model for comparison config."""

    general: ComparisonGeneral
    preconditioners: tuple[PreconditionerConfig, ...]


class _SolverParamsModel(BaseModel):
    rtol: float = Field(default=DEFAULT_RTOL, gt=0.0)
    atol: float = Field(default=DEFAULT_ATOL, ge=0.0)
    max_iterations: int = Field(default=100, ge=1)
    stopping_criterion: Literal["residual_norm", "fixed_iterations"] = "residual_norm"
    m_max: int = Field(default=DEFAULT_M_MAX, ge=-1)
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonDataModel(BaseModel):
    """Parser model for ``[general.data]`` inside a comparison method file.

    ``matrix_path`` and ``rhs_path`` are absent here — they are injected by
    ``resolve_comparison_config`` from the case ``[[comparisons]]`` entry.
    """

    matrix_path: Path | None = None
    rhs_path: Path | None = None
    matrix_index: int = 0
    dataset_alias: str | None = None
    normalize_system: NormalizeSystem = "matrix"
    require_non_residual_rhs: bool = True
    selection_seed: int | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("matrix_path", "rhs_path", mode="before")
    @classmethod
    def _expand_paths(cls, v: object, info: ValidationInfo) -> object:
        if not isinstance(v, str) or info.context is None:
            return v
        from neuralls.platform.config.context import expand_config_path

        ctx = ConfigContext.from_pydantic_context(info.context)
        return expand_config_path(v, ctx)


class _ComparisonGeneralModel(BaseModel):
    params: _SolverParamsModel | None = None
    data: ComparisonDataModel = Field(default_factory=ComparisonDataModel)
    model_config = ConfigDict(extra="forbid", frozen=True)


class GaussianRhsSourceModel(BaseModel):
    """Direct Gaussian RHS source for comparison workflows."""

    kind: Literal["gaussian"] = "gaussian"
    mean: float = 0.0
    std: float = Field(default=1.0, gt=0.0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class SparseRhsSourceModel(BaseModel):
    """Direct sparse RHS source for comparison workflows."""

    kind: Literal["sparse"] = "sparse"
    indices: list[int]
    values: list[float]

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _validate_lengths(self) -> SparseRhsSourceModel:
        if len(self.indices) != len(self.values):
            raise ValueError("Sparse RHS source requires indices and values of equal length.")
        return self


def _parse_row_kind(value: object) -> RowKind | None:
    """Parse row-kind config values from enum, integer code, or lowercase name."""
    if value is None or isinstance(value, RowKind):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for kind in RowKind:
            if normalized in {kind.name.lower(), str(kind.value)}:
                return kind
    if isinstance(value, int):
        return RowKind(value)
    raise ValueError(f"Unsupported row_kind value: {value!r}")


class _RawVectorSourceBase(BaseModel):
    """Shared fields for concrete raw vector comparison sources."""

    path: Path
    sample_index: int = Field(default=0, ge=0)
    row_kind: RowKind | None = None
    scale: float = 1.0

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("path", mode="before")
    @classmethod
    def _expand_path(cls, v: object, info: ValidationInfo) -> object:
        if not isinstance(v, str) or info.context is None:
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @field_validator("row_kind", mode="before")
    @classmethod
    def _parse_row_kind(cls, v: object) -> RowKind | None:
        return _parse_row_kind(v)


class RawLhsSourceModel(_RawVectorSourceBase):
    """Load a raw solution-side vector and convert it to RHS with A @ vector."""

    kind: Literal["raw_lhs"] = "raw_lhs"


class RawRhsSourceModel(_RawVectorSourceBase):
    """Load a raw vector that is already an RHS."""

    kind: Literal["raw_rhs"] = "raw_rhs"


class DatasetRhsSourceModel(BaseModel):
    """Load a comparison RHS from a manifest-backed neuralls dataset directory."""

    kind: Literal["dataset"] = "dataset"
    path: Path
    sample_index: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("path", mode="before")
    @classmethod
    def _expand_path(cls, v: object, info: ValidationInfo) -> object:
        if not isinstance(v, str) or info.context is None:
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))


type ComparisonRhsSourceModel = Annotated[
    GaussianRhsSourceModel
    | SparseRhsSourceModel
    | RawLhsSourceModel
    | RawRhsSourceModel
    | DatasetRhsSourceModel,
    Field(discriminator="kind"),
]


class _ComparisonConfigModel(BaseModel):
    """Parser for comparison method override files.

    All sections are optional — absent sections inherit from case
    ``[comparison_defaults]`` when resolved via ``resolve_comparison_config``.
    """

    general: _ComparisonGeneralModel = Field(default_factory=_ComparisonGeneralModel)
    preconditioners: list[PreconditionerConfig] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_preconditioners(self) -> _ComparisonConfigModel:
        """Validate any preconditioners that are present."""
        if not self.preconditioners:
            return self
        names = [spec.name for spec in self.preconditioners]
        if len(names) != len(set(names)):
            raise ValueError("Comparison config cannot define duplicate preconditioner names.")
        neural_specs = [
            spec for spec in self.preconditioners if isinstance(spec, NeuralPreconditionerConfig)
        ]
        if not neural_specs:
            return self

        for spec in neural_specs:
            if spec.model_ref is None:
                raise ValueError(f"Neural preconditioner '{spec.name}' must define model_ref.")
            if spec.checkpoint_path is not None:
                raise ValueError(
                    f"Neural preconditioner '{spec.name}' cannot use checkpoint_path in comparison configs."
                )
            if (
                isinstance(spec.model_ref, RegisteredModelRefConfig)
                and spec.model_ref.name is None
                and spec.assignment is None
            ):
                raise ValueError(
                    f"Neural preconditioner '{spec.name}' must define assignment when "
                    "registered model_ref.name is omitted."
                )

        requires_dataset_alias = any(
            isinstance(spec.model_ref, RegisteredModelRefConfig)
            and spec.model_ref.alias is not None
            and spec.model_ref.alias.strip() == "@dataset"
            and spec.assignment is None
            for spec in neural_specs
        )
        if requires_dataset_alias and self.general.data.dataset_alias is None:
            raise ValueError(
                "general.data.dataset_alias is required when model_ref alias is '@dataset' "
                "and no neural preconditioner assignment binding is provided."
            )
        return self


def parse_comparison_config(
    raw: dict[str, object],
    context: ConfigContext,
) -> ComparisonConfig:
    """Parse a standalone comparison TOML into a ComparisonConfig domain model.

    This function is for comparison files that include explicit ``matrix_path``,
    ``rhs_path``, solver params, and preconditioners (legacy / inference use).
    For case-driven comparisons use ``resolve_comparison_config`` in the registry.

    Args:
        raw: Deserialized TOML dict for the comparison config file.
        context: ConfigContext for expanding ${NEURALLS_*} path placeholders.

    Returns:
        Fully validated ComparisonConfig domain model.

    Raises:
        ValueError: If required fields (matrix_path, rhs_path, preconditioners,
            or solver params) are absent.
    """
    parsed = _ComparisonConfigModel.model_validate(raw, context=context.as_pydantic_context())
    data = parsed.general.data
    params = parsed.general.params
    if data.matrix_path is None:
        raise ValueError("Standalone comparison config must specify [general.data].matrix_path.")
    if data.rhs_path is None:
        raise ValueError("Standalone comparison config must specify [general.data].rhs_path.")
    if params is None:
        raise ValueError("Standalone comparison config must specify [general.params].")
    if not parsed.preconditioners:
        raise ValueError(
            "Standalone comparison config must define at least one [[preconditioners]] entry."
        )
    return ComparisonConfig(
        general=ComparisonGeneral(
            params=SolverParams(**params.model_dump()),
            data=ComparisonData(**data.model_dump()),
        ),
        preconditioners=tuple(parsed.preconditioners),
    )


def parse_comparison_method_override(
    raw: dict[str, object],
    context: ConfigContext,
) -> _ComparisonConfigModel:
    """Parse a comparison method override file (partial config, no paths required).

    Returns the raw parsed model so callers can merge it with case defaults.

    Args:
        raw: Deserialized TOML dict for the method override file.
        context: ConfigContext for expanding ${NEURALLS_*} path placeholders.

    Returns:
        Parsed partial config model for merging with case comparison_defaults.
    """
    return _ComparisonConfigModel.model_validate(raw, context=context.as_pydantic_context())
