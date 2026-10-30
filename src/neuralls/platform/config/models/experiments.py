"""Pydantic models for the top-level case config."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from neuralls.platform.config.context import ConfigContext, expand_config_path
from neuralls.platform.config.models.comparison import ComparisonRhsSourceModel
from neuralls.platform.config.models.id_generation import (
    _infer_assignment_id,
    _infer_comparison_id,
)
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.config.models.presentation import ScientificLabelsConfig
from neuralls.shared.constants import DEFAULT_ATOL, DEFAULT_M_MAX, DEFAULT_RTOL
from neuralls.shared.scientific_description.models import ValidatedLabel
from neuralls.shared.timing import TimingRepeatPolicy


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
        report: Name for the experiment holding reports derived from comparison sessions.
    """

    training: str = "Train"
    comparison: str = "Comparisons"
    report: str = "Reports"
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

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))


class JobEntry(RegistryEntry):
    """A job registry entry; ``display_name`` is its training title, ``architecture`` its model name.

    Attributes:
        architecture: Short plain name of the network this job trains (e.g. ``"Graph Net"``),
            shown in comparison labels. Separate from ``display_name`` because a training
            title ("... Optuna Search (ReLU)") is long and free text, while a label must be
            a short controlled name. Required only when the job feeds a neural comparison.
    """

    architecture: str | None = None

    @model_validator(mode="after")
    def _check_architecture_is_a_plain_name(self) -> JobEntry:
        """Reject an architecture that could not be shown in every render target."""
        if self.architecture is not None:
            try:
                ValidatedLabel(self.architecture)
            except ValueError as exc:
                raise ValueError(f"Job '{self.id}' architecture: {exc}") from exc
        return self

    @property
    def architecture_label(self) -> ValidatedLabel | None:
        """The architecture parsed into its validated form."""
        return None if self.architecture is None else ValidatedLabel(self.architecture)


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

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))


class ComparisonBaseline(BaseModel):
    """Shared comparison methodology every ``[[comparisons]]`` entry starts from.

    Every field already carries its own source-code default (``DEFAULT_RTOL``
    and friends below), so ``[comparison_baseline]`` is never a required full
    declaration — a case states only the fields it wants to diverge from the
    source default, and a per-entry ``method`` file may in turn diverge from
    this baseline. "Baseline" names that two-layer relationship explicitly,
    since "defaults" alone is ambiguous between "this model's own field
    defaults" and "the case-wide baseline these fields establish."

    Attributes:
        rtol: Relative convergence tolerance.
        atol: Absolute convergence tolerance.
        max_iterations: Maximum solver iterations.
        stopping_criterion: Convergence check strategy.
        m_max: FCG orthogonalization window.
        normalize_system: Normalization applied to the test system.
        preconditioners: Classical preconditioner list (neural preconditioners are
            auto-generated from case assignments at runtime).
        reference_preconditioner: Optional explicit method used as the paired
            comparison reference; omitted means no paired reference report.
        presentation: Scientific-label notation for this case's preconditioner
            comparison plots (target: plain/mathtext/latex, symbol overrides).
            Complete defaults, so no case needs a ``[comparison_baseline.presentation]``
            table at all. A per-entry ``method`` file may override it the same
            way it overrides ``preconditioners``/``normalize_system``.
        timing: How often each preconditioner build and each solve is timed
            (``repeats``, ``warmup``, ``statistic``). Defaults to one timed run;
            raise ``repeats`` for steadier estimates when systems are cheap
            enough to afford it. Capped by ``MAX_TIMING_REPEATS``.
    """

    rtol: float = Field(default=DEFAULT_RTOL, gt=0.0)
    atol: float = Field(default=DEFAULT_ATOL, ge=0.0)
    max_iterations: int = Field(default=100, ge=1)
    stopping_criterion: Literal["residual_norm", "fixed_iterations"] = "residual_norm"
    m_max: int = Field(default=DEFAULT_M_MAX, ge=-1)
    normalize_system: Literal["none", "matrix", "rhs", "both"] = "matrix"
    preconditioners: list[PreconditionerConfig] = Field(default_factory=list)
    reference_preconditioner: str | None = None
    presentation: ScientificLabelsConfig = Field(default_factory=ScientificLabelsConfig)
    timing: TimingRepeatPolicy = Field(default_factory=TimingRepeatPolicy)
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonDataBinding(BaseModel):
    """Fields shared by every comparison data-binding entry, sweep or not.

    Both ``ComparisonRegistryEntry`` (single-matrix dataset) and
    ``ComparisonSweepEntry`` (multi-matrix dataset) describe the same data binding — which matrix dataset, which RHS
    source, which method override, which assignments to compare. Sharing this
    base keeps those fields from drifting apart between the two shapes.

    Attributes:
        matrix_dataset: Dataset id (from ``[[datasets]]``) used as the test matrix.
        method: Optional path to a comparison TOML that overrides defaults.
        rhs_source: RHS source specification (`gaussian`, `sinusoidal`, raw file, or `dataset`), each optionally masked.
        rhs_count: How many RHS draws to solve per matrix.
        require_non_residual_rhs: Reject RHS rows flagged as solver-internal residuals.
        seed: Optional base seed; each planned system derives its own RHS seed from it.
        assignments: Optional assignment id filter; empty means all assignments.
        display_name: Optional human-facing label.
    """

    matrix_dataset: str
    method: Path | None = None
    rhs_source: ComparisonRhsSourceModel
    rhs_count: int = Field(default=1, ge=1)
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

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))


class ComparisonRegistryEntry(ComparisonDataBinding):
    """Comparison registry entry: data binding + optional methodology override.

    The test matrix and RHS are identified by dataset IDs from the case
    ``[[datasets]]`` registry.  Solver parameters and the base preconditioner
    list are inherited from ``[comparison_baseline]`` unless a ``method`` file
    is given to override them. The dataset must hold a single matrix (see
    ``ComparisonSweepEntry`` for multi-matrix datasets).

    Attributes:
        id: Stable comparison identifier.
    """

    id: str = Field(..., min_length=1)


class FractionSelector(BaseModel):
    """Randomly sample a fraction of a dataset's held-out matrices.

    ``p = 1.0`` selects every held-out matrix.
    """

    kind: Literal["fraction"] = "fraction"
    p: float = Field(..., gt=0.0, le=1.0)
    seed: int

    model_config = ConfigDict(extra="forbid", frozen=True)


class FirstNSelector(BaseModel):
    """Select the first ``n`` held-out matrices of a dataset, in ascending matrix-id order."""

    kind: Literal["first_n"] = "first_n"
    n: int = Field(..., gt=0)

    model_config = ConfigDict(extra="forbid", frozen=True)


type TestSetSelector = Annotated[
    FractionSelector | FirstNSelector,
    Field(discriminator="kind"),
]


class ComparisonSweepEntry(ComparisonDataBinding):
    """One sweep over a multi-matrix dataset's held-out matrices.

    A sweep entry never names a matrix itself — ``selector`` chooses among the
    matrices of whichever dataset's pinned test fold ``matrix_dataset``
    identifies, and the planner expands the entry into one system per
    (selected matrix, RHS draw).

    Attributes:
        selector: Which held-out matrices this sweep covers.
        id_prefix: Stem used to build each planned system's id, as
            ``f"{id_prefix}-{matrix_id}"`` (plus ``-{k}`` when ``rhs_count > 1``).
    """

    selector: TestSetSelector
    id_prefix: str = Field(..., min_length=1)


def _dedupe_ids(kind: str, entries: Sequence[RegistryEntry | ComparisonRegistryEntry]) -> None:
    """Reject duplicate registry ids with a focused error."""
    counts = Counter(entry.id for entry in entries)
    duplicates = sorted(id_ for id_, count in counts.items() if count > 1)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate {kind} ids in case config: {joined}.")


def _dedupe_assignment_ids(entries: list[AssignmentEntry]) -> None:
    """Reject duplicate assignment ids with a focused error."""
    counts = Counter(entry.id for entry in entries)
    duplicates = sorted(id_ for id_, count in counts.items() if count > 1)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate assignment ids in case config: {joined}.")


def _registry_ids(entries: Sequence[RegistryEntry]) -> set[str]:
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
        comparison_sweeps: Sweep entries, each expanded at batch time into
            many ``ComparisonRegistryEntry`` objects via a ``TestSetSelector``.
        assignments: Assignment entries referencing registry ids.
        mlflow: MLflow topology config (tracking URI, etc.).
        names: MLflow experiment names for training and comparison.
        comparison_baseline: Shared solver params and preconditioner list applied to
            all comparisons unless overridden by a per-entry ``method`` file.
    """

    datasets: list[RegistryEntry] = Field(default_factory=list)
    jobs: list[JobEntry] = Field(default_factory=list)
    comparisons: list[ComparisonRegistryEntry] = Field(default_factory=list)
    comparison_sweeps: list[ComparisonSweepEntry] = Field(default_factory=list)
    assignments: list[AssignmentEntry] = Field(default_factory=list)
    mlflow: MlflowTopologyConfig = Field(default_factory=MlflowTopologyConfig)
    names: ExperimentNamesConfig = Field(default_factory=ExperimentNamesConfig)
    comparison_baseline: ComparisonBaseline | None = None

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _auto_fill_ids_and_display_names(cls, data: object) -> object:
        """Auto-generate missing ids before child validation.

        A missing ``display_name`` stays absent: it is an explicit editorial title or
        nothing, never a label rebuilt from registry ids.

        Args:
            data: Raw input dict (or other type, passed through unchanged).

        Returns:
            The mutated dict with any missing id fields filled in,
            or the original data if it is not a dict.

        Raises:
            ValueError: If any inferred or user-supplied id contains invalid characters,
                or if a display_name cannot be slugified into a valid id.
        """
        if not isinstance(data, dict):
            return data
        raw = cast("dict[str, object]", data)

        assignments: list[object] = cast("list[object]", raw.get("assignments", []))
        comparisons: list[object] = cast("list[object]", raw.get("comparisons", []))

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

        for comp in comparisons:
            if not isinstance(comp, dict):
                continue
            comp_dict = cast("dict[str, object]", comp)
            matrix_id = str(comp_dict.get("matrix_dataset") or "")
            rhs_source = comp_dict.get("rhs_source")
            rhs_kind = rhs_source.get("kind") if isinstance(rhs_source, dict) else None
            rhs_id = str(rhs_kind or "")
            # A masked source is a different problem from the unmasked one of the
            # same kind, so entries declaring both must not collide on one id.
            mask = rhs_source.get("mask") if isinstance(rhs_source, dict) else None
            if isinstance(mask, dict) and mask.get("kind") != "none":
                rhs_id = f"{rhs_id}-masked"
            raw_id = comp_dict.get("id")
            raw_dn = comp_dict.get("display_name")
            user_id = str(raw_id).strip() if isinstance(raw_id, str) else None
            user_id = user_id or None
            user_dn = str(raw_dn).strip() if isinstance(raw_dn, str) else None
            user_dn = user_dn or None

            comp_dict["id"] = _infer_comparison_id(matrix_id, rhs_id, user_id, user_dn)

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
