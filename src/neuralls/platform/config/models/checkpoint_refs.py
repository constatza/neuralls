"""Checkpoint references: how a config names the trained network it loads."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from neuralls.platform.config.context import ConfigContext, expand_config_path
from neuralls.platform.config.models.fact_roles import IDENTITY_ROLE
from neuralls.platform.config.models.slugs import slug_text
from neuralls.shared.digest import Cosmetic, InputConfig, InputData


class ByAlias(BaseModel):
    """Registered model selected by alias.

    Attributes:
        alias: Model alias (e.g. ``"@solutions"``); the ``@`` prefix is stripped on resolution.
    """

    alias: str
    model_config = ConfigDict(extra="forbid", frozen=True)


class ByVersion(BaseModel):
    """Registered model selected by an explicit version number."""

    version: int = Field(ge=1)
    model_config = ConfigDict(extra="forbid", frozen=True)


class LatestVersion(BaseModel):
    """Registered model selected as its highest version number.

    This is "most recently registered," not "best-performing" - there is no
    metric-based (e.g. lowest validation loss) selection. Use an explicit
    alias pointed at a deliberately-chosen version for a stable,
    quality-vetted reference.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


type RegistrySelector = ByAlias | ByVersion | LatestVersion

_REGISTRY_SELECTOR_KEYS = ("alias", "version", "latest")


class RegisteredModelRefConfig(BaseModel):
    """Reference to a model registered in the MLflow Model Registry.

    TOML and keyword construction stay flat (``alias = ...``, ``version = ...``
    or ``latest = true``); the before-validator folds them into ``selector``
    so downstream code matches on one closed sum type.

    Attributes:
        source: Discriminator field, always "registry".
        name: Registered model name when explicitly provided. Only meaningful
            for a registered model with no assignment linkage at all (e.g. an
            externally-trained baseline this system never trained). When a
            ``NeuralPreconditionerConfig.assignment`` is set, the registered
            model name is derived automatically from the assignment; do not
            also set `name` in that case (see
            ``NeuralPreconditionerConfig.validate_single_model_identity_source``).
        selector: Which registered version to use.
    """

    source: Literal["registry"] = "registry"
    name: str | None = Field(default=None, min_length=1)
    selector: RegistrySelector
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _fold_flat_selector(cls, data: object) -> object:
        """Fold the flat ``alias`` / ``version`` / ``latest`` fields into ``selector``.

        Raises:
            ValueError: If not exactly one of alias, version, or latest=true is set.
        """
        if not isinstance(data, dict) or "selector" in data:
            return data
        chosen: list[RegistrySelector] = []
        if data.get("alias") is not None:
            chosen.append(ByAlias(alias=data["alias"]))
        if data.get("version") is not None:
            chosen.append(ByVersion(version=data["version"]))
        if data.get("latest") is True:
            chosen.append(LatestVersion())
        if len(chosen) != 1:
            raise ValueError("Requires exactly one selector: alias, version, or latest=true.")
        rest = {k: v for k, v in data.items() if k not in _REGISTRY_SELECTOR_KEYS}
        return {**rest, "selector": chosen[0]}

    def slug(self) -> str:
        """Deterministic identity segment for this reference."""
        match self.selector:
            case ByAlias(alias=alias):
                selector = alias
            case ByVersion(version=version):
                selector = f"v{version}"
            case LatestVersion():
                selector = "latest"
        return f"registry_{slug_text(self.name or 'assignment')}_{slug_text(selector)}"


class ByRunId(BaseModel):
    """Logged model selected by an explicit MLflow run id."""

    run_id: str = Field(min_length=1)
    model_config = ConfigDict(extra="forbid", frozen=True)


class LatestMatching(BaseModel):
    """Logged model selected as the most recently *started* run matching the filters.

    Runs are ordered by ``attributes.start_time DESC``. This is "most recent,"
    not "best-performing" - there is no metric-based selection.

    Attributes:
        model_name: Filter by logged model name.
        experiment_name: Filter by experiment name.
        experiment_id: Filter by experiment ID.
        run_name: Filter by run name.
        tags: Tag filters.
    """

    model_name: str | None = None
    experiment_name: str | None = None
    experiment_id: str | None = None
    run_name: str | None = None
    tags: dict[str, str] | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _require_a_filter(self) -> LatestMatching:
        """Reject a ``latest`` selection that would match every run."""
        filters = (self.model_name, self.experiment_name, self.experiment_id, self.run_name)
        if self.tags is None and all(f is None for f in filters):
            raise ValueError(
                "latest=true requires at least one filter "
                "(model_name, experiment_name, experiment_id, run_name, or tags)."
            )
        return self


type LoggedSelector = ByRunId | LatestMatching

_LOGGED_FILTER_KEYS = ("model_name", "experiment_name", "experiment_id", "run_name", "tags")


class LoggedModelRefConfig(BaseModel):
    """Reference to a model logged within an MLflow run.

    TOML and keyword construction stay flat (``run_id = ...`` or ``latest =
    true`` plus filters); the before-validator folds them into ``selector``.

    Attributes:
        source: Discriminator field, always "run".
        selector: Explicit run id, or the latest run matching filters.
        artifact_path: Artifact path within the run (default: "model").
    """

    source: Literal["run"] = "run"
    selector: LoggedSelector
    artifact_path: str = "model"
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _fold_flat_selector(cls, data: object) -> object:
        """Fold the flat ``run_id`` / ``latest`` / filter fields into ``selector``.

        Raises:
            ValueError: If not exactly one of run_id or latest=true is set, or
                filters accompany an explicit run_id.
        """
        if not isinstance(data, dict) or "selector" in data:
            return data
        has_run_id = data.get("run_id") is not None
        is_latest = data.get("latest") is True
        if has_run_id == is_latest:
            raise ValueError("Requires exactly one selector: run_id or latest=true.")
        filters = {k: data[k] for k in _LOGGED_FILTER_KEYS if data.get(k) is not None}
        rest = {
            k: v for k, v in data.items() if k not in ("run_id", "latest", *_LOGGED_FILTER_KEYS)
        }
        if has_run_id:
            if filters:
                raise ValueError("run_id cannot be combined with filters; use latest=true.")
            return {**rest, "selector": ByRunId(run_id=data["run_id"])}
        return {**rest, "selector": LatestMatching(**filters)}

    def slug(self) -> str:
        """Deterministic identity segment for this reference."""
        match self.selector:
            case ByRunId(run_id=run_id):
                return f"run_{slug_text(run_id)}"
            case LatestMatching(model_name=model_name):
                return f"run_latest_{slug_text(model_name or 'unset')}"


class TrainedAssignmentRefConfig(BaseModel):
    """Reference to the training run that produced an assignment's current model.

    Unlike ``LoggedModelRefConfig(latest=True, ...)`` this is not a tag search:
    it resolves to exactly one run that is (a) in the case's training
    experiment, (b) tagged with this assignment id, (c) FINISHED with a real
    checkpoint, and (d) trained on the dataset as it exists on disk right now
    (matching ``dataset_hash``). A run from another experiment, a failed or
    incomplete run, or a run trained on a since-regenerated dataset is never
    selected. The experiment and dataset hash come from the resolution
    context, not from this config.

    Attributes:
        source: Discriminator field, always "assignment".
        assignment_id: Stable assignment id the training run was tagged with.
    """

    source: Literal["assignment"] = "assignment"
    assignment_id: str = Field(min_length=1)
    model_config = ConfigDict(extra="forbid", frozen=True)

    def slug(self) -> str:
        """Deterministic identity segment for this reference."""
        return f"assignment_{slug_text(self.assignment_id)}"


ModelRefConfig = Annotated[
    RegisteredModelRefConfig | LoggedModelRefConfig | TrainedAssignmentRefConfig,
    Field(discriminator="source"),
]


class NeuralCheckpointRef(BaseModel):
    """Shared checkpoint-identity fields for any config that loads a trained network.

    Composition (a nested ``identity`` field) would be the usual preference for
    an orthogonal concern like this, but Pydantic composition would force a
    nested TOML sub-table, breaking the existing flat shape of configs that
    already embed these fields. Inheritance keeps the flat shape. Consumers
    that only need to resolve a checkpoint should type against this mixin
    directly rather than against a concrete leaf class.
    """

    checkpoint_path: Annotated[Path | None, InputData(), IDENTITY_ROLE] = None
    assignment: Annotated[str | None, IDENTITY_ROLE] = None
    config_path: Annotated[Path | None, InputConfig(), IDENTITY_ROLE] = None
    data_config_path: Annotated[Path | None, InputConfig(), IDENTITY_ROLE] = None
    model_ref: Annotated[ModelRefConfig | None, IDENTITY_ROLE] = None
    resolved_checkpoint_path: Annotated[Path | None, Cosmetic(), IDENTITY_ROLE] = None
    resolved_run_id: Annotated[str | None, Cosmetic(), IDENTITY_ROLE] = None
    """MLflow run id the checkpoint was resolved from, when resolved via
    `model_ref` (unset for an explicit `checkpoint_path`). Used to detect when
    a comparison's dependency has been retrained since it last ran."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    @property
    def active_checkpoint_path(self) -> Path | None:
        """The checkpoint to load: the resolved MLflow artifact when present, else the explicit path."""
        return self.resolved_checkpoint_path or self.checkpoint_path

    def checkpoint_slug(self) -> str:
        """Deterministic slug for whichever checkpoint-identity source this ref actually sets.

        Priority: ``assignment`` / ``model_ref`` (the stable, *logical* reference)
        over a bare ``checkpoint_path``. Checkpoint resolution
        (``model_resolution.py``) fills in ``checkpoint_path`` with the resolved
        *local* path once a ``model_ref`` is resolved - a path that can differ run
        to run (temp dirs, cache locations) even for the same logical reference.
        Following it would give a re-run of the same comparison a different MLflow
        metric-key segment. A bare ``checkpoint_path`` with no
        ``model_ref``/``assignment`` (an explicit local file, never resolved from
        MLflow) has nothing better to key on, so it is the fallback.
        """
        if self.assignment is not None:
            return f"assignment_{slug_text(self.assignment)}"
        if self.model_ref is not None:
            return self.model_ref.slug()
        if self.checkpoint_path is not None:
            return f"path_{slug_text(self.checkpoint_path.stem)}"
        return "unset"

    @field_validator("checkpoint_path", "config_path", "data_config_path", mode="before")
    @classmethod
    def _expand_paths(cls, v: object, info: ValidationInfo) -> object:
        """Expand ${NEURALLS_*} placeholders and resolve checkpoint-related paths.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if v is None or info.context is None or not isinstance(v, str):
            return v

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))


@runtime_checkable
class CheckpointRefBearing(Protocol):
    """A preconditioner config that owns one or more resolvable checkpoint refs.

    The resolution layer (``composition/assignments/model_resolution.py``)
    resolves every ``NeuralCheckpointRef`` reachable from a preconditioner
    spec the same way, regardless of whether the config *is* a ref
    (``NeuralPreconditionerConfig``) or *has* one nested in a field
    (``AMGPreconditionerConfig.coarsening``, ``NeuralAMGPreconditionerConfig
    .prolongation``/``.restriction``) — implementing this protocol is what
    makes a config participate, with no branching on ``PreconditionerType``
    in the resolver itself.

    Invariant: ``with_resolved_refs`` must accept a tuple of exactly the
    labels and length last returned by ``checkpoint_refs()`` — the resolver
    always round-trips the same set it read, never a different arity.
    """

    def checkpoint_refs(self) -> tuple[tuple[str, NeuralCheckpointRef], ...]:
        """Return every ``(label, ref)`` pair reachable from this config."""
        ...

    def with_resolved_refs(self, resolved: tuple[tuple[str, NeuralCheckpointRef], ...]) -> Self:
        """Rebuild this config with the given resolved refs substituted in."""
        ...
