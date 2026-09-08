"""Pydantic models for the top-level case config."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from neuralls.platform.config.models.comparison import ComparisonRhsSourceModel
from neuralls.platform.config.models.id_generation import (
    _build_display_lookup,
    _infer_assignment_display_name,
    _infer_assignment_id,
    _infer_comparison_display_name,
    _infer_comparison_id,
)
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.shared.constants import DEFAULT_ATOL, DEFAULT_M_MAX, DEFAULT_RTOL


def resolve_display_name(entity_id: str, display_name: str | None) -> str:
    """Return the human-facing label for a registry entry."""
    if display_name is None:
        return entity_id
    stripped = display_name.strip()
    return stripped or entity_id


class MlflowTopologyConfig(BaseModel):
    """MLflow topology configuration.

    Attributes:
        tracking_uri: MLflow tracking URI.
        artifacts_destination: Optional artifacts root for local sqlite tracking.
    """

    tracking_uri: str | None = None
    artifacts_destination: str | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("tracking_uri", "artifacts_destination", mode="before")
    @classmethod
    def _expand_if_placeholder(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Expand ${NEURALLS_*} placeholders only; plain URIs pass through unchanged.

        Args:
            v: Raw string value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Expanded path string if v contains a placeholder, otherwise v unchanged.
        """
        if v is None or info.context is None or "${" not in v:
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @model_validator(mode="after")
    def validate_topology(self) -> MlflowTopologyConfig:
        """Reject ambiguous partial topology definitions."""
        if self.artifacts_destination is not None and self.tracking_uri is None:
            raise ValueError(
                "Case config [mlflow] cannot set artifacts_destination without tracking_uri."
            )
        return self


class SharedTrackingSettings(BaseModel):
    """Shared dlkit-native tracking settings from configs/tracking.toml [tracking] section.

    Attributes:
        backend: dlkit tracking backend identifier ("mlflow" or "none").
        uri: MLflow server URI forwarded to dlkit TrackingSettings.
        artifacts_destination: Optional artifact root for local backends (neuralls only).
    """

    backend: str = "mlflow"
    uri: str | None = None
    artifacts_destination: str | None = None


class ExperimentNamesConfig(BaseModel):
    """MLflow experiment names configuration.

    Attributes:
        training: Name for training experiments.
        comparison: Name for comparison experiments.
    """

    training: str = "Train"
    comparison: str = "Comparisons"
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegistryEntry(BaseModel):
    """Single registry entry with an explicit config path.

    Attributes:
        id: Stable lookup id used by the case config registry.
        path: Relative or absolute config path.
        display_name: Optional human-facing label.
    """

    id: str = Field(..., min_length=1)
    path: Path
    display_name: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("path", mode="before")
    @classmethod
    def _expand_path(cls, v: object, info: ValidationInfo) -> object:
        """Expand ${NEURALLS_*} placeholders and resolve to absolute path.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if info.context is None or not isinstance(v, str):
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @property
    def effective_display_name(self) -> str:
        """Return the configured label or fall back to the id."""
        return resolve_display_name(self.id, self.display_name)


class AssignmentEntry(BaseModel):
    """Single assignment entry from the ``[[assignments]]`` case table.

    An assignment pairs one job with one dataset — the intent "run this job
    on this dataset." Not to be confused with an MLflow Experiment (a bucket
    grouping many runs) — see ``ExperimentNamesConfig`` for that concept.

    Attributes:
        id: Stable assignment identifier.
        dataset_id: Dataset registry id.
        job_id: Job registry id.
        display_name: Optional human-facing label.
    """

    id: str = Field(..., min_length=1)
    dataset_id: str = Field(..., alias="dataset")
    job_id: str = Field(..., alias="job")
    checkpoint_path: Path | None = None
    display_name: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def _expand_checkpoint_path(cls, v: object, info: ValidationInfo) -> object:
        """Expand ${NEURALLS_*} placeholders and resolve checkpoint path.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if v is None or info.context is None or not isinstance(v, str):
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @property
    def effective_display_name(self) -> str:
        """Return the configured label or fall back to the id."""
        return resolve_display_name(self.id, self.display_name)


class ComparisonDefaults(BaseModel):
    """Shared methodology defaults applied to all ``[[comparisons]]`` entries.

    Attributes:
        rtol: Relative convergence tolerance.
        atol: Absolute convergence tolerance.
        max_iterations: Maximum solver iterations.
        stopping_criterion: Convergence check strategy.
        m_max: FCG orthogonalization window.
        normalize_system: Normalization applied to the test system.
        preconditioners: Classical preconditioner list (neural preconditioners are
            auto-generated from case assignments at runtime).
    """

    rtol: float = Field(default=DEFAULT_RTOL, gt=0.0)
    atol: float = Field(default=DEFAULT_ATOL, ge=0.0)
    max_iterations: int = Field(default=100, ge=1)
    stopping_criterion: Literal["residual_norm", "fixed_iterations"] = "residual_norm"
    m_max: int = Field(default=DEFAULT_M_MAX, ge=-1)
    normalize_system: Literal["none", "matrix", "rhs", "both"] = "matrix"
    preconditioners: list[PreconditionerConfig] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonRegistryEntry(BaseModel):
    """Comparison registry entry: data binding + optional methodology override.

    The test matrix and RHS are identified by dataset IDs from the case
    ``[[datasets]]`` registry.  Solver parameters and the base preconditioner
    list are inherited from ``[comparison_defaults]`` unless a ``method`` file
    is given to override them.

    Attributes:
        id: Stable comparison identifier.
        matrix_dataset: Dataset id (from ``[[datasets]]``) used as the test matrix.
        matrix_index: Matrix sample index when the dataset stores multiple matrices.
            Defaults to 0.
        method: Optional path to a comparison TOML that overrides defaults.
        rhs_source: RHS source specification (`gaussian`, `sparse`, raw file, or `dataset`).
        assignments: Optional assignment id filter; empty means all assignments.
        display_name: Optional human-facing label.
    """

    id: str = Field(..., min_length=1)
    matrix_dataset: str
    matrix_index: int = 0
    method: Path | None = None
    rhs_source: ComparisonRhsSourceModel
    require_non_residual_rhs: bool = True
    seed: int | None = None
    assignments: list[str] = Field(default_factory=list)
    display_name: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("method", mode="before")
    @classmethod
    def _expand_method_path(cls, v: object, info: ValidationInfo) -> object:
        """Expand ${NEURALLS_*} placeholders in the method file path.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if v is None or not isinstance(v, str) or info.context is None:
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @property
    def effective_display_name(self) -> str:
        """Return the configured label or fall back to the id."""
        return resolve_display_name(self.id, self.display_name)

    @model_validator(mode="after")
    def _validate_rhs_source(self) -> ComparisonRegistryEntry:
        if self.matrix_index < 0:
            raise ValueError("matrix_index must be non-negative when provided.")
        return self


def _dedupe_ids(kind: str, entries: Sequence[RegistryEntry | ComparisonRegistryEntry]) -> None:
    """Reject duplicate registry ids with a focused error."""
    seen: set[str] = set()
    duplicates = sorted({entry.id for entry in entries if entry.id in seen or seen.add(entry.id)})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate {kind} ids in case config: {joined}.")


def _dedupe_assignment_ids(entries: list[AssignmentEntry]) -> None:
    """Reject duplicate assignment ids with a focused error."""
    seen: set[str] = set()
    duplicates = sorted({entry.id for entry in entries if entry.id in seen or seen.add(entry.id)})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate assignment ids in case config: {joined}.")


def _registry_ids(entries: list[RegistryEntry]) -> set[str]:
    """Return registry ids for fast membership checks."""
    return {entry.id for entry in entries}


def _validate_comparison_assignment_filter_refs(
    comparisons: list[ComparisonRegistryEntry],
    assignment_ids: set[str],
) -> None:
    """Reject comparison assignment filter refs that reference unknown assignment ids."""
    for entry in comparisons:
        unknown = [aid for aid in entry.assignments if aid not in assignment_ids]
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(
                f"Comparison '{entry.id}' assignments filter references unknown "
                f"assignment ids: {joined}."
            )


def _validate_comparison_dataset_refs(
    comparisons: list[ComparisonRegistryEntry],
    *,
    dataset_ids: set[str],
) -> None:
    """Reject comparison entries that reference undefined dataset ids."""
    for entry in comparisons:
        if entry.matrix_dataset not in dataset_ids:
            raise ValueError(
                f"Comparison '{entry.id}' references matrix_dataset "
                f"'{entry.matrix_dataset}', but [[datasets]] does not define it."
            )


def _validate_assignment_registry_refs(
    assignments: list[AssignmentEntry],
    *,
    dataset_ids: set[str],
    job_ids: set[str],
) -> None:
    """Reject assignments that reference undefined dataset/job registry ids."""
    for entry in assignments:
        if entry.dataset_id not in dataset_ids:
            raise ValueError(
                f"Assignment '{entry.id}' references dataset id "
                f"'{entry.dataset_id}', but [[datasets]] does not define it."
            )
        if entry.job_id not in job_ids:
            raise ValueError(
                f"Assignment '{entry.id}' references job id "
                f"'{entry.job_id}', but [[jobs]] does not define it."
            )


class CaseConfig(BaseModel):
    """Top-level case configuration.

    Attributes:
        datasets: Dataset registry entries (training data + comparison reference data).
        jobs: Job registry entries.
        comparisons: Comparison registry entries with data binding.
        assignments: Assignment entries referencing registry ids.
        mlflow: MLflow topology config (tracking URI, etc.).
        names: MLflow experiment names for training and comparison.
        comparison_defaults: Shared solver params and preconditioner list applied to
            all comparisons unless overridden by a per-entry ``method`` file.
    """

    datasets: list[RegistryEntry] = Field(default_factory=list)
    jobs: list[RegistryEntry] = Field(default_factory=list)
    comparisons: list[ComparisonRegistryEntry] = Field(default_factory=list)
    assignments: list[AssignmentEntry] = Field(default_factory=list)
    mlflow: MlflowTopologyConfig = Field(default_factory=MlflowTopologyConfig)
    names: ExperimentNamesConfig = Field(default_factory=ExperimentNamesConfig)
    comparison_defaults: ComparisonDefaults | None = None

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _auto_fill_ids_and_display_names(cls, data: object) -> object:
        """Auto-generate missing ids and display names before child validation.

        Args:
            data: Raw input dict (or other type, passed through unchanged).

        Returns:
            The mutated dict with any missing id/display_name fields filled in,
            or the original data if it is not a dict.

        Raises:
            ValueError: If any inferred or user-supplied id contains invalid characters,
                or if a display_name cannot be slugified into a valid id.
        """
        if not isinstance(data, dict):
            return data
        raw = cast("dict[str, object]", data)

        datasets: list[object] = cast("list[object]", raw.get("datasets", []))
        jobs: list[object] = cast("list[object]", raw.get("jobs", []))
        assignments: list[object] = cast("list[object]", raw.get("assignments", []))
        comparisons: list[object] = cast("list[object]", raw.get("comparisons", []))

        dataset_display = _build_display_lookup(datasets)
        model_display = _build_display_lookup(jobs)

        for assignment in assignments:
            if not isinstance(assignment, dict):
                continue
            assignment_dict = cast("dict[str, object]", assignment)
            dataset_id = str(assignment_dict.get("dataset") or "")
            job_id = str(assignment_dict.get("job") or "")
            raw_id = assignment_dict.get("id")
            raw_dn = assignment_dict.get("display_name")
            user_id = str(raw_id).strip() if isinstance(raw_id, str) else None
            user_id = user_id or None
            user_dn = str(raw_dn).strip() if isinstance(raw_dn, str) else None
            user_dn = user_dn or None

            assignment_dict["id"] = _infer_assignment_id(dataset_id, job_id, user_id, user_dn)

            if not user_dn:
                assignment_dict["display_name"] = _infer_assignment_display_name(
                    dataset_id, job_id, user_dn, dataset_display, model_display
                )

        for comp in comparisons:
            if not isinstance(comp, dict):
                continue
            comp_dict = cast("dict[str, object]", comp)
            matrix_id = str(comp_dict.get("matrix_dataset") or "")
            rhs_source = comp_dict.get("rhs_source")
            rhs_kind = rhs_source.get("kind") if isinstance(rhs_source, dict) else None
            rhs_id = str(rhs_kind or "")
            raw_id = comp_dict.get("id")
            raw_dn = comp_dict.get("display_name")
            user_id = str(raw_id).strip() if isinstance(raw_id, str) else None
            user_id = user_id or None
            user_dn = str(raw_dn).strip() if isinstance(raw_dn, str) else None
            user_dn = user_dn or None

            comp_dict["id"] = _infer_comparison_id(matrix_id, rhs_id, user_id, user_dn)

            if not user_dn:
                comp_dict["display_name"] = _infer_comparison_display_name(
                    matrix_id,
                    rhs_id,
                    user_dn,
                    dataset_display,
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_case_config_tables(cls, data: object) -> object:
        """Reject unsupported case-config table names."""
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        if "experiment" in raw:
            raise ValueError(
                "Unsupported '[[experiment]]' table. Use '[[assignments]]' entries in case config."
            )
        if "experiments" in raw:
            raise ValueError(
                "Unsupported '[[experiments]]' table. Use '[[assignments]]' entries in case config."
            )
        if "models" in raw:
            raise ValueError(
                "Unsupported '[[models]]' table. Use '[[jobs]]' entries in case config."
            )
        if "comparison_profiles" in raw:
            raise ValueError(
                "Unsupported 'comparison_profiles' table. "
                "Use [[comparisons]] entries in case config."
            )
        return raw

    @model_validator(mode="after")
    def validate_unique_ids(self) -> CaseConfig:
        """Reject duplicate ids and validate all cross-registry references."""
        _dedupe_ids("dataset registry", self.datasets)
        _dedupe_ids("job registry", self.jobs)
        _dedupe_ids("comparison registry", self.comparisons)
        _dedupe_assignment_ids(self.assignments)
        dataset_ids = _registry_ids(self.datasets)
        _validate_assignment_registry_refs(
            self.assignments,
            dataset_ids=dataset_ids,
            job_ids=_registry_ids(self.jobs),
        )
        _validate_comparison_assignment_filter_refs(
            self.comparisons,
            assignment_ids={a.id for a in self.assignments},
        )
        _validate_comparison_dataset_refs(self.comparisons, dataset_ids=dataset_ids)
        return self
