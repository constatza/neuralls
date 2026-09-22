"""Preconditioner configuration models.

These Pydantic models validate preconditioner configurations from TOML files.
They support both factory creation and scheduling/comparison concerns.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, cast, runtime_checkable

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from neuralls.shared.digest import Cosmetic, InputConfig, InputData


class PreconditionerType(StrEnum):
    """Preconditioner types."""

    NONE = "none"
    IDENTITY = "identity"
    JACOBI = "jacobi"
    ILU = "ilu"
    IC0 = "ic0"
    ICHOLESKY = "icholesky"
    NEURAL = "neural"
    AMG = "amg"
    NEURAL_AMG = "neural_amg"
    ADAPTIVE_SA_AMG = "adaptive_sa_amg"


def _normalize_null(data: dict) -> Any:
    """Normalize deprecated null-like aliases to the explicit none baseline."""
    if isinstance(data, dict):
        data = data.copy()
        if data.get("type") == "null":
            data["type"] = PreconditionerType.NONE
        return data
    return data


class RegisteredModelRefConfig(BaseModel):
    """Reference to a model registered in the MLflow Model Registry.

    Attributes:
        source: Discriminator field, always "registry".
        name: Registered model name when explicitly provided. Only meaningful
            for a registered model with no assignment linkage at all (e.g. an
            externally-trained baseline this system never trained). When a
            ``NeuralPreconditionerConfig.assignment`` is set, the registered
            model name is derived automatically from the assignment; do not
            also set `name` in that case (see
            ``NeuralPreconditionerConfig.validate_single_model_identity_source``).
        alias: Model alias (e.g. ``"@solutions"``); ``@`` prefix is stripped.
        version: Explicit model version number.
        latest: If True, select the highest registered version number. This is
            "most recently registered," not "best-performing" — there is no
            metric-based (e.g. lowest validation loss) selection. Use an
            explicit `alias` pointed at a deliberately-chosen version if you
            need a stable, quality-vetted reference.
    """

    source: Literal["registry"] = "registry"
    name: str | None = Field(default=None, min_length=1)
    alias: str | None = None
    version: int | None = Field(default=None, ge=1)
    latest: bool | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_selector(self) -> RegisteredModelRefConfig:
        """Validate that exactly one selector is provided.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If not exactly one of alias, version, or latest=True is set.
        """
        selectors = [self.alias is not None, self.version is not None, self.latest is True]
        if sum(selectors) != 1:
            raise ValueError("Requires exactly one selector: alias, version, or latest=true.")
        return self


class LoggedModelRefConfig(BaseModel):
    """Reference to a model logged within an MLflow run.

    Attributes:
        source: Discriminator field, always "run".
        run_id: Explicit MLflow run ID to reference.
        latest: If True, select the most recently *started* run matching the
            given filters (ordered by ``attributes.start_time DESC``). This is
            "most recent," not "best-performing" — there is no metric-based
            (e.g. lowest validation loss) selection.
        model_name: Filter by logged model name.
        experiment_name: Filter by experiment name.
        experiment_id: Filter by experiment ID.
        run_name: Filter by run name.
        artifact_path: Artifact path within the run (default: "model").
        tags: Optional tag filters.
    """

    source: Literal["run"] = "run"
    run_id: str | None = Field(default=None, min_length=1)
    latest: bool | None = None
    model_name: str | None = None
    experiment_name: str | None = None
    experiment_id: str | None = None
    run_name: str | None = None
    artifact_path: str = "model"
    tags: dict[str, str] | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_selector(self) -> LoggedModelRefConfig:
        """Validate that exactly one selection mode is active.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If not exactly one of run_id or latest=True is set.
        """
        selectors = [self.run_id is not None, self.latest is True]
        if sum(selectors) != 1:
            raise ValueError("Requires exactly one selector: run_id or latest=true.")
        return self

    @model_validator(mode="after")
    def validate_latest_filters(self) -> LoggedModelRefConfig:
        """Validate that latest=True is accompanied by at least one filter.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If latest=True but no filter fields are set.
        """
        if self.latest is not True:
            return self
        filters = [
            self.model_name,
            self.experiment_name,
            self.experiment_id,
            self.run_name,
            self.tags,
        ]
        if not any(f is not None for f in filters):
            raise ValueError(
                "latest=true requires at least one filter "
                "(model_name, experiment_name, experiment_id, run_name, or tags)."
            )
        return self


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

    checkpoint_path: Annotated[Path | None, InputData()] = None
    assignment: str | None = None
    config_path: Annotated[Path | None, InputConfig()] = None
    data_config_path: Annotated[Path | None, InputConfig()] = None
    model_ref: ModelRefConfig | None = None
    resolved_checkpoint_path: Annotated[Path | None, Cosmetic()] = None
    resolved_run_id: Annotated[str | None, Cosmetic()] = None
    """MLflow run id the checkpoint was resolved from, when resolved via
    `model_ref` (unset for an explicit `checkpoint_path`). Used to detect when
    a comparison's dependency has been retrained since it last ran."""

    model_config = ConfigDict(extra="ignore", frozen=True)

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
        from neuralls.platform.config.context import ConfigContext, expand_config_path

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


class BasePreconditionerConfig(BaseModel):
    """Shared fields for all preconditioners.

    Includes scheduling fields shared by all preconditioner variants.
    """

    name: Annotated[str, Cosmetic()] = Field(
        default="",
        description="Config identity, used as the results/plot key. Defaults to "
        "the concrete subclass's `type` when omitted.",
    )
    start_iter: int = Field(
        default=0,
        ge=0,
        description="Iteration at which the primary preconditioner becomes active.",
    )
    limit_iters: int = Field(default=-1, description="Iterations to apply; -1 means unlimited.")
    fallback: PreconditionerType = Field(
        default=PreconditionerType.IDENTITY,
        description="Fallback preconditioner type when limited.",
    )

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields from conversion
        frozen=True,
    )

    @model_validator(mode="after")
    def _default_name_to_type(self) -> Self:
        """Fall back `name` to the concrete subclass's `type` value when omitted.

        Returns:
            The validated config instance.
        """
        if not self.name:
            object.__setattr__(self, "name", str(getattr(self, "type", "")))
        return self


class StandardPreconditionerConfig(BasePreconditionerConfig):
    """Non-parametric, static preconditioners (identity, jacobi, ilu, icholesky)."""

    type: Literal[
        PreconditionerType.NONE,
        PreconditionerType.IDENTITY,
        PreconditionerType.JACOBI,
        PreconditionerType.ILU,
        PreconditionerType.ICHOLESKY,
    ]


class IC0PreconditionerConfig(BasePreconditionerConfig):
    """IC(0) preconditioner configuration with threshold parameter."""

    type: Literal[PreconditionerType.IC0] = PreconditionerType.IC0
    threshold: float = Field(
        default=0.0,
        description=(
            "Drop tolerance - entries with |value| <= threshold are treated as zeros. "
            "An absolute constant is unsafe here: comparisons run against a "
            "normalize_system-scaled matrix, so a fixed absolute threshold's "
            "effective strictness shifts with each matrix's own scale and can "
            "silently drop structurally-real entries, corrupting the IC(0) "
            "sparsity pattern. Default 0.0 only drops exact zeros."
        ),
    )


class NeuralPreconditionerConfig(BasePreconditionerConfig, NeuralCheckpointRef):
    """Neural preconditioner configuration."""

    type: Literal[PreconditionerType.NEURAL] = PreconditionerType.NEURAL
    extra_input_names: tuple[str, ...] = Field(
        default=(),
        description=(
            "Names of extra dataset arrays to bind before CG, beyond the residual. "
            "E.g. ('matrix',) for the stiffness matrix, ('coordinates',) for node coords. "
            "The comparison workflow loads matching named arrays from the dataset directory."
        ),
    )

    @field_validator("extra_input_names", mode="before")
    @classmethod
    def _coerce_extra_names_to_tuple(cls, v: object) -> tuple[str, ...] | object:
        """Coerce list to tuple for extra_input_names field.

        Args:
            v: Raw value that may be list, tuple, or other type.

        Returns:
            Tuple of strings, or original value if not list/tuple.
        """
        if isinstance(v, (list, tuple)):
            return tuple(str(x) for x in v)
        return v

    def checkpoint_refs(self) -> tuple[tuple[str, NeuralCheckpointRef], ...]:
        """This preconditioner's own identity is its single checkpoint ref."""
        return (("", self),)

    def with_resolved_refs(
        self, resolved: tuple[tuple[str, NeuralCheckpointRef], ...]
    ) -> NeuralPreconditionerConfig:
        """Resolution replaces this spec's own identity fields."""
        _, ref = resolved[0]
        return cast(NeuralPreconditionerConfig, ref)

    @model_validator(mode="after")
    def validate_single_model_identity_source(self) -> NeuralPreconditionerConfig:
        """Reject overlapping registered-model identity sources.

        `assignment` derives the registered model name automatically
        (`build_registered_model_name(assignment_id)`); `model_ref.name` on a
        `RegisteredModelRefConfig` names a registered model unrelated to any
        assignment. Setting both would leave undocumented, silent precedence
        between two competing identity sources.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If both `assignment` and a `RegisteredModelRefConfig`
                `model_ref.name` are set.
        """
        if (
            self.assignment is not None
            and isinstance(self.model_ref, RegisteredModelRefConfig)
            and self.model_ref.name is not None
        ):
            raise ValueError(
                "Set either 'assignment' (which derives the registered model name "
                "automatically) or 'model_ref.name' (for a registered model unrelated "
                "to any assignment) — not both; assignment is the single source of "
                "truth for an assignment's registered model identity."
            )
        return self


class AggregationCoarseningConfig(BaseModel):
    """Classical smoothed-aggregation coarsening (SA-AMG)."""

    method: Literal["aggregation"] = "aggregation"
    theta: float = Field(
        default=0.25,
        gt=0.0,
        lt=1.0,
        description="Strength-of-connection threshold controlling AMG aggregate/coarse-grid size.",
    )
    omega: float | None = Field(
        default=None,
        gt=0.0,
        description="Prolongation Jacobi-smoothing damping; auto per-matrix via "
        "torchalg's spectral-radius rule when unset.",
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    def exposed_checkpoint_ref(self) -> NeuralCheckpointRef | None:
        """Purely algebraic coarsening — never checkpoint-backed."""
        return None


class TargetDimCoarseningConfig(BaseModel):
    """AMG-aggregation coarsening parameterized by target coarse dimension, not theta.

    `theta` is a strength-of-connection threshold; the coarse dimension it
    produces is emergent, not chosen, and (confirmed against real stiffness
    matrices, see `docs/plan.md`) a step function of `theta` rather than a
    smooth or monotonic one. This gives AMG-aggregation the same "set the
    coarse dimension directly" ergonomics `PODCoarseningConfig.rank` already
    has, via an adaptive `theta` search for the closest realized match
    (`torchalg.preconditioners.implementations.amg.TargetDimensionCoarsening`
    — a wrapper external to `AggregationCoarsening`/`AMGPreconditioner`,
    never modifying either; `composition/preconditioners/factory.py`
    constructs it with `cache_candidates=True` to share the winning
    theta's full build across sibling `target_dim` configs against the
    same matrix within one comparison run).
    """

    method: Literal["target_dim"] = "target_dim"
    target_coarse_dim: int = Field(gt=0, description="Desired realized coarse dimension.")
    theta_min: float = Field(default=0.01, gt=0.0, lt=1.0, description="Lower theta search bound.")
    theta_max: float = Field(default=0.99, gt=0.0, lt=1.0, description="Upper theta search bound.")
    step: float = Field(default=0.01, gt=0.0, description="Theta search grid spacing.")
    omega: float | None = Field(
        default=None,
        gt=0.0,
        description="Prolongation Jacobi-smoothing damping; auto per-matrix via "
        "torchalg's spectral-radius rule when unset.",
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    def exposed_checkpoint_ref(self) -> NeuralCheckpointRef | None:
        """Purely algebraic coarsening — never checkpoint-backed."""
        return None


class RawWeightingConfig(BaseModel):
    """No snapshot weighting — every snapshot votes equally in the POD SVD.

    The default `PODCoarseningConfig.weighting`, reproducing the exact
    behavior POD-2G had before per-snapshot weighting existed.
    """

    method: Literal["raw"] = "raw"
    model_config = ConfigDict(extra="forbid", frozen=True)


class PowerNormWeightingConfig(BaseModel):
    """Row scale `||e_k||^(-beta)`, interpolating between raw and normalized snapshots.

    `beta=0` is `RawWeightingConfig` in disguise (every snapshot keeps its
    own magnitude); `beta=1` fully normalizes each snapshot by `metric`
    before the SVD — `"l2"` for `e_k / ||e_k||_2`, `"a"` for the A-inner-product
    `e_k / ||e_k||_A`. Values in between interpolate. See
    `torchalg.preconditioners.implementations.pod.weighting.power_norm_scales`,
    which this config resolves to.
    """

    method: Literal["power_norm"] = "power_norm"
    metric: Literal["l2", "a"] = "l2"
    beta: float = Field(default=1.0, ge=0.0, description="Normalization exponent.")
    model_config = ConfigDict(extra="forbid", frozen=True)


class SmootherPersistenceWeightingConfig(BaseModel):
    """Row scale by how well each snapshot survives weighted-Jacobi damping.

    Weights each snapshot by `||G^steps e_k|| / ||e_k||`, `G = I - omega D^-1 A`
    — the smoother's own error-propagation operator — so snapshots the
    smoother already removes efficiently contribute little to the fitted
    basis, and smoother-resistant ("algebraically smooth") snapshots
    contribute most. `omega` defaults to unset, matching
    `torchalg.preconditioners.implementations.amg.smoothers.JacobiSmoother`'s
    own default of auto-computing the damping per-matrix via its
    spectral-radius rule, so this targets the same smoother POD-2G actually
    runs alongside. See
    `torchalg.preconditioners.implementations.pod.weighting.smoother_persistence_scales`,
    which this config resolves to.
    """

    method: Literal["smoother_persistence"] = "smoother_persistence"
    omega: float | None = Field(
        default=None,
        gt=0.0,
        description="Weighted-Jacobi damping factor; auto per-matrix via "
        "torchalg's spectral-radius rule when unset.",
    )
    steps: int = Field(default=5, gt=0, description="Number of damping sweeps.")
    model_config = ConfigDict(extra="forbid", frozen=True)


SnapshotWeightingConfig = Annotated[
    RawWeightingConfig | PowerNormWeightingConfig | SmootherPersistenceWeightingConfig,
    Field(discriminator="method"),
]

parse_snapshot_weighting_config = TypeAdapter(SnapshotWeightingConfig).validate_python
"""Coerce a raw dict (as a TOML `[model]` table's `extra="allow"` passthrough
yields — see `composition/preconditioners/pod_fittable.py`) or an already-typed
instance into a validated `SnapshotWeightingConfig` member."""


class PODCoarseningConfig(NeuralCheckpointRef):
    """POD-2G coarsening (Nikolopoulos et al. 2022, §3.3-3.5).

    The prolongation/restriction operator is a POD basis fit to a snapshot
    ensemble of high-fidelity solutions (or, for a sharper coarse space,
    CG error traces e_k = x* - x_k, which are richer than raw solution vectors
    since e_0 == x* and later k emphasize slow-converging directions) —
    read from an already-generated dataset directory, not raw files, so the
    same validation/normalization/manifest guarantees as every other
    dataset in this repo apply.

    Inherits the optional checkpoint-identity fields from
    ``NeuralCheckpointRef`` (``assignment``, ``model_ref``, ``checkpoint_path``,
    ``resolved_checkpoint_path``) so a POD-2G basis can *also* be fit once
    ahead of time via a ``FitJobConfig`` assignment and reconstructed from its
    MLflow-tracked checkpoint at comparison time, instead of being refit
    inline from ``dataset_dir`` on every run — see
    ``composition/preconditioners/factory.py``'s AMG branch. None of these
    fields are required: a `PODCoarseningConfig` with no checkpoint identity
    set behaves exactly as before (inline `dataset_dir` fit only).
    """

    method: Literal["pod"] = "pod"
    dataset_dir: Annotated[Path, InputData()] = Field(
        ...,
        description="Generated dataset directory whose `solutions` array supplies POD-2G snapshots.",
    )
    n_snapshots: int = Field(
        default=-1, description="Number of snapshot files to load; -1 means all matched."
    )
    rank: int | float = Field(
        default=8,
        description=(
            "Fixed number of POD modes to retain (int), or minimum cumulative "
            "captured energy to retain (float in (0, 1] — e.g. 0.9999)."
        ),
    )
    weighting: SnapshotWeightingConfig = Field(
        default_factory=RawWeightingConfig,
        description=(
            "Per-snapshot row scaling applied before the POD SVD — "
            "`RawWeightingConfig` (default) reproduces the original unweighted "
            "fit exactly; see `SnapshotWeightingConfig`'s member configs."
        ),
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("dataset_dir", mode="before")
    @classmethod
    def _expand_dataset_dir(cls, v: object, info: ValidationInfo) -> object:
        """Expand ``${NEURALLS_*}`` placeholders and anchor the dataset directory.

        Args:
            v: Raw field value.
            info: Pydantic validation context.

        Returns:
            Resolved path string, or original value if not a string.
        """
        from neuralls.platform.config.context import expand_path_field

        return expand_path_field(v, info)

    @field_validator("rank")
    @classmethod
    def _validate_rank(cls, v: float) -> int | float:
        """Enforce the per-mode constraint matching whichever branch was matched.

        Args:
            v: The parsed ``rank`` value — an int (mode count) or float
                (energy threshold).

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If an int rank is < 1, or a float rank is outside (0, 1].
        """
        if isinstance(v, float):
            if not (0.0 < v <= 1.0):
                raise ValueError(f"rank as an energy threshold must be in (0, 1], got {v}")
        elif v < 1:
            raise ValueError(f"rank as a mode count must be >= 1, got {v}")
        return v

    def exposed_checkpoint_ref(self) -> NeuralCheckpointRef | None:
        """Opt into checkpoint resolution only when a resolvable identity is set.

        Unlike a neural preconditioner (always checkpoint-backed), POD
        coarsening is equally valid unfitted-from-scratch (``dataset_dir`` +
        ``rank`` only, no checkpoint fields at all) — that case must stay
        invisible to `CheckpointRefBearing` resolution so it keeps falling
        through to today's inline fit unchanged.
        """
        has_identity = (
            self.assignment is not None
            or self.model_ref is not None
            or self.checkpoint_path is not None
        )
        return self if has_identity else None


class NeuralPODCoarseningConfig(NeuralCheckpointRef):
    """POD-2G coarsening whose snapshot ensemble is predicted by a checkpoint.

    Same POD-2G math as ``PODCoarseningConfig`` (Nikolopoulos et al. 2022,
    §3.3-3.5), but the high-fidelity snapshot ensemble is *predicted* by a
    trained network instead of read from precomputed solutions: ``dataset_dir``
    supplies the network's input parameter arrays (not the solutions
    themselves), and the checkpoint identity fields inherited from
    ``NeuralCheckpointRef`` locate the model that turns those parameters into
    snapshots.
    """

    method: Literal["neural_pod"] = "neural_pod"
    dataset_dir: Annotated[Path, InputData()] = Field(
        ...,
        description="Generated dataset directory whose `params` arrays feed the checkpoint.",
    )
    input_names: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Checkpoint forward()/predict() keyword-argument names, one per "
            "dataset `params` array, in the same order. These are declared by "
            "the checkpoint's DLKit training job — NOT derived from how the "
            "dataset writer happened to name the on-disk params artifacts; "
            "there is no relationship between the two, so this must be set "
            "explicitly (mirrors NeuralPreconditionerConfig.extra_input_names)."
        ),
    )
    n_snapshots: int = Field(
        default=-1, description="Number of predicted snapshots to keep; -1 means all."
    )
    rank: int | float = Field(
        default=8,
        description=(
            "Fixed number of POD modes to retain (int), or minimum cumulative "
            "captured energy to retain (float in (0, 1] — e.g. 0.9999)."
        ),
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("dataset_dir", mode="before")
    @classmethod
    def _expand_dataset_dir(cls, v: object, info: ValidationInfo) -> object:
        """Expand ``${NEURALLS_*}`` placeholders and anchor the dataset directory.

        Args:
            v: Raw field value.
            info: Pydantic validation context.

        Returns:
            Resolved path string, or original value if not a string.
        """
        from neuralls.platform.config.context import expand_path_field

        return expand_path_field(v, info)

    @field_validator("input_names", mode="before")
    @classmethod
    def _coerce_input_names_to_tuple(cls, v: object) -> tuple[str, ...] | object:
        """Coerce list to tuple for input_names field.

        Args:
            v: Raw value that may be list, tuple, or other type.

        Returns:
            Tuple of strings, or original value if not list/tuple.
        """
        if isinstance(v, (list, tuple)):
            return tuple(str(x) for x in v)
        return v

    @field_validator("rank")
    @classmethod
    def _validate_rank(cls, v: float) -> int | float:
        """Enforce the per-mode constraint matching whichever branch was matched.

        Args:
            v: The parsed ``rank`` value — an int (mode count) or float
                (energy threshold).

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If an int rank is < 1, or a float rank is outside (0, 1].
        """
        if isinstance(v, float):
            if not (0.0 < v <= 1.0):
                raise ValueError(f"rank as an energy threshold must be in (0, 1], got {v}")
        elif v < 1:
            raise ValueError(f"rank as a mode count must be >= 1, got {v}")
        return v

    def exposed_checkpoint_ref(self) -> NeuralCheckpointRef:
        """Always checkpoint-backed — its snapshot ensemble only exists via a checkpoint."""
        return self


CoarseningConfig = Annotated[
    AggregationCoarseningConfig
    | PODCoarseningConfig
    | NeuralPODCoarseningConfig
    | TargetDimCoarseningConfig,
    Field(discriminator="method"),
]


class AMGPreconditionerConfig(BasePreconditionerConfig):
    """AMG-family preconditioner configuration (multigrid coarsening + cycle).

    ``coarsening`` selects the strategy that builds the prolongation/
    restriction operator — ``AggregationCoarseningConfig`` (classical SA-AMG),
    ``PODCoarseningConfig`` (POD-2G), or ``NeuralPODCoarseningConfig``
    (checkpoint-predicted POD-2G). Required, no default: the caller must
    state which coarsening strategy an AMG config means. A future coarsening
    strategy is added the same way: one more class in the
    ``CoarseningConfig`` union, not a new ``PreconditionerType`` and not new
    fields on this class — the underlying ``AMGPreconditioner`` is already
    generic over any ``CoarseningStrategy``.
    """

    type: Literal[PreconditionerType.AMG] = PreconditionerType.AMG
    n_levels: int = Field(default=2, ge=2, description="Total number of grid levels.")
    pre_smoothing_steps: int = Field(default=2, ge=0, description="Pre-smoothing iterations.")
    post_smoothing_steps: int = Field(default=2, ge=0, description="Post-smoothing iterations.")
    smoother_omega: float | None = Field(
        default=None,
        gt=0.0,
        description="Weighted Jacobi damping; auto per-matrix via torchalg's "
        "spectral-radius rule when unset.",
    )
    coarsening: CoarseningConfig

    def checkpoint_refs(self) -> tuple[tuple[str, NeuralCheckpointRef], ...]:
        """Expose the coarsening's checkpoint ref, when it carries one.

        Each coarsening class answers for itself via ``exposed_checkpoint_ref``
        — a new coarsening strategy declares its own answer rather than adding
        a branch here.
        """
        ref = self.coarsening.exposed_checkpoint_ref()
        return (("coarsening", ref),) if ref is not None else ()

    def with_resolved_refs(
        self, resolved: tuple[tuple[str, NeuralCheckpointRef], ...]
    ) -> AMGPreconditionerConfig:
        """Rebuild with a resolved coarsening ref substituted in."""
        _, ref = resolved[0]
        return self.model_copy(update={"coarsening": cast(CoarseningConfig, ref)})


class AdaptiveSAPreconditionerConfig(BasePreconditionerConfig):
    """Adaptive smoothed-aggregation AMG preconditioner (alpha-SA, PyAMG-faithful).

    Unlike ``AMGPreconditionerConfig``, this is not a pluggable-coarsening
    variant of the shared ``AMGPreconditioner(cycle=VCycle(JacobiSmoother))``
    path: ``torchalg.preconditioners.implementations.amg.AdaptiveSAPreconditioner``
    builds its whole multilevel hierarchy eagerly at construction (learned
    near-null-space candidates, symmetric Gauss-Seidel V(1,1) cycle, a
    pseudo-inverse coarse solve) and cannot be composed with
    ``AggregationCoarsening``/``TargetDimensionCoarsening`` the way POD-2G or
    classical SA-AMG can — hence its own ``PreconditionerType`` rather than
    another ``CoarseningConfig`` union member.

    ``theta`` should stay small for aSA — it is not the tuning knob for
    coarse-space size here, ``num_candidates`` is (see below). PyAMG's own
    default, independently confirmed against PyAMG's real source
    (``pyamg/aggregation/adaptive.py``'s ``adaptive_sa_solver`` ->
    ``pyamg/strength.py``'s ``symmetric_strength_of_connection(A, theta=0)``),
    is ``theta=0.0`` — "every non-zero coupling is strong", i.e. *maximal*
    aggregation/smallest coarse space, not "big": raising ``theta`` prunes
    the strength graph and makes the coarse space *larger*. There is no
    target-coarse-dimension search here (unlike classical SA-AMG's
    ``TargetDimCoarseningConfig``) — a real end-to-end run showed the
    search picking a *worse* theta than just naming one directly (its
    classical-AMG target-dim=10 search landed on a fully degenerate
    ``c=0``, while fixed ``theta=0.05`` alone gives ``c=23`` on the same
    504x504 matrix, see ``docs/plan.md``) — so pick ``theta`` directly
    instead, and keep it close to ``0`` (this repo's default is ``0.05``,
    the same value used at every tier; do not raise it to grow the coarse
    space — see ``num_candidates`` below).

    ``num_candidates`` is the actual coarse-space-size knob for aSA, not
    ``theta``: it defaults to ``3``, not PyAMG's own default of ``1``
    (PyAMG ships the cheapest, least-adaptive starting point and expects
    callers to raise it; its own examples typically use 2-4). Each
    additional candidate adds a DOF to every aggregate, so realized coarse
    dimension scales as ``(aggregate count at theta) x num_candidates`` —
    measured on a real 504x504 stiffness matrix from this repo
    (``SpectralData/45x15/stiffness/subdomain_1_Kaa.txt``) at a fixed
    ``theta=0.05``: ``c`` goes 69 (``k=3``) -> 115 (``k=5``) -> 161
    (``k=7``), so raising ``num_candidates`` alone reaches the same coarse
    dimension this repo's classical SA-AMG reaches by raising ``theta`` to
    ``0.25`` (also ``c=160``) — without ever leaving aSA's small-``theta``,
    maximal-aggregation regime, and while adding real multi-candidate
    adaptivity (more independent near-null-space directions) instead of
    just cruder aggregation.
    """

    type: Literal[PreconditionerType.ADAPTIVE_SA_AMG] = PreconditionerType.ADAPTIVE_SA_AMG
    n_levels: int = Field(
        default=2, ge=2, description="Cap on hierarchy depth (torchalg's `max_levels`)."
    )
    num_candidates: int = Field(
        default=3,
        ge=1,
        description="Total near-null-space candidates (including the initial one); "
        "PyAMG's own default is 1, but each extra candidate adds real adaptivity "
        "at the cost of one more DOF per aggregate in the realized coarse dimension.",
    )
    candidate_iters: int = Field(
        default=5, ge=1, description="Relaxation sweeps/cycles per candidate step."
    )
    max_coarse: int = Field(
        default=10, ge=1, description="Stop coarsening once a level has this many nodes or fewer."
    )
    theta: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Strength-of-connection threshold — pick directly, no search; "
        "close to 0 for a small coarse space, ~0.25 for a larger one.",
    )
    omega: float = Field(
        default=4.0 / 3.0, gt=0.0, description="Nominal Jacobi prolongator-smoothing damping."
    )
    seed: int = Field(
        default=0, description="Seed for the candidate/spectral-radius random source."
    )


class NeuralTransferConfig(NeuralCheckpointRef):
    """Config for one neural transfer operator (prolongation or restriction).

    Required inputs (the extra arrays the network expects beyond its vector input)
    are NOT declared here — they are read at runtime from
    ``ExtraInputPredictorPort.required_inputs``, which is populated by the adapter
    from the model config.  This eliminates duplication with the DLKit TOML.
    """


class NeuralAMGPreconditionerConfig(BasePreconditionerConfig):
    """AMG preconditioner that uses neural networks for prolongation/restriction.

    The ``extra_input_names`` required by each network are NOT declared here —
    they are read at runtime from ``ExtraInputPredictorPort.required_inputs`` so
    that the DLKit model config remains the single source of truth.
    """

    type: Literal[PreconditionerType.NEURAL_AMG] = PreconditionerType.NEURAL_AMG
    n_levels: int = Field(default=2, ge=2, description="Total number of grid levels.")
    pre_smoothing_steps: int = Field(default=2, ge=0)
    post_smoothing_steps: int = Field(default=2, ge=0)
    smoother_omega: float | None = Field(default=None, gt=0.0)
    aggregation_omega: float | None = Field(default=None, gt=0.0)
    prolongation: NeuralTransferConfig
    restriction: NeuralTransferConfig | None = None

    def checkpoint_refs(self) -> tuple[tuple[str, NeuralCheckpointRef], ...]:
        """Expose prolongation's ref, plus restriction's when set."""
        refs: tuple[tuple[str, NeuralCheckpointRef], ...] = (("prolongation", self.prolongation),)
        if self.restriction is not None:
            refs += (("restriction", self.restriction),)
        return refs

    def with_resolved_refs(
        self, resolved: tuple[tuple[str, NeuralCheckpointRef], ...]
    ) -> NeuralAMGPreconditionerConfig:
        """Rebuild with resolved prolongation/restriction refs substituted in."""
        updates: dict[str, NeuralTransferConfig] = {}
        for label, ref in resolved:
            updates[label] = cast(NeuralTransferConfig, ref)
        return self.model_copy(update=updates)


ConcretePreconditionerConfig = (
    StandardPreconditionerConfig
    | IC0PreconditionerConfig
    | NeuralPreconditionerConfig
    | AMGPreconditionerConfig
    | NeuralAMGPreconditionerConfig
    | AdaptiveSAPreconditionerConfig
)

_StrictPreconditionerConfig = Annotated[
    ConcretePreconditionerConfig,
    Field(discriminator="type"),
]

PreconditionerConfig = Annotated[
    _StrictPreconditionerConfig,
    BeforeValidator(_normalize_null),
]

parse_preconditioner_config = TypeAdapter(PreconditionerConfig).validate_python
