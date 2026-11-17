"""AMG coarsening configuration models (algebraic and POD-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationInfo,
    field_validator,
)

from neuralls.platform.config.context import expand_path_field
from neuralls.platform.config.models.checkpoint_refs import NeuralCheckpointRef
from neuralls.platform.config.models.fact_roles import (
    IDENTITY_ROLE,
    OPERATIONAL_ROLE,
    PROVENANCE_ROLE,
)
from neuralls.platform.config.models.slugs import slug_number, slug_text
from neuralls.shared.digest import InputData
from neuralls.shared.fact_binding import (
    Fact,
    Show,
    enum_value,
)
from neuralls.shared.scientific_description.catalog import (
    DAMPING,
    INNER_PRODUCT,
    RANK,
    SNAPSHOT_COUNT,
    TARGET_COARSE_DIMENSION,
    THRESHOLD,
)
from neuralls.shared.types import InnerProductKind

ALL_SNAPSHOTS = -1
"""The ``n_snapshots`` value that keeps every snapshot."""

type SnapshotCount = Literal[-1] | PositiveInt
"""A snapshot cut: ``ALL_SNAPSHOTS``, or a positive count; zero and other negatives are invalid."""


class AggregationCoarseningConfig(BaseModel):
    """Classical smoothed-aggregation coarsening (SA-AMG).

    Complexity: setup is one strength-of-connection pass plus greedy
    aggregation, O(nnz(A)) on a truly sparse matrix — this pipeline's dense
    tensor storage (see ``IC0PreconditionerConfig``'s complexity note for
    the same caveat) makes the practical cost closer to O(n^2) since forming
    and scanning the dense strength graph touches every entry.
    """

    method: Annotated[Literal["aggregation"], IDENTITY_ROLE] = "aggregation"
    theta: Annotated[float, Fact(THRESHOLD)] = Field(
        default=0.25,
        gt=0.0,
        lt=1.0,
        description="Strength-of-connection threshold controlling AMG aggregate/coarse-grid size.",
    )
    omega: Annotated[float | None, Fact(DAMPING, show=Show.NOT_NONE)] = Field(
        default=None,
        gt=0.0,
        description="Prolongation Jacobi-smoothing damping; auto per-matrix via "
        "torchalg's spectral-radius rule when unset.",
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    def exposed_checkpoint_ref(self) -> NeuralCheckpointRef | None:
        """Purely algebraic coarsening — never checkpoint-backed."""
        return None

    def slug(self) -> str:
        """Deterministic slug for this coarsening."""
        omega_text = slug_number(self.omega) if self.omega is not None else "auto"
        return f"aggregation_theta{slug_number(self.theta)}_omega{omega_text}"


class TargetDimCoarseningConfig(BaseModel):
    """AMG-aggregation coarsening parameterized by target coarse dimension, not theta.

    `theta` is a strength-of-connection threshold; the coarse dimension it
    produces is emergent, not chosen, and (confirmed against real stiffness
    matrices) a step function of `theta` rather than a
    smooth or monotonic one. This gives AMG-aggregation the same "set the
    coarse dimension directly" ergonomics `PODCoarseningConfig.rank` already
    has, via an adaptive `theta` search for the closest realized match
    (`torchalg.preconditioners.implementations.amg.TargetDimensionCoarsening`
    — a wrapper external to `AggregationCoarsening`/`AMGPreconditioner`,
    never modifying either; `composition/preconditioners/factory.py`
    constructs it with `cache_candidates=True` to share the winning
    theta's full build across sibling `target_dim` configs against the
    same matrix within one comparison run).

    Complexity: `AggregationCoarseningConfig`'s setup cost (see its own
    complexity note), multiplied by however many `theta` values the grid
    search in `(theta_min, theta_max, step)` tries before landing on the
    closest match — `cache_candidates=True` (this repo's factory always sets
    it) means repeated `target_dim` configs against the same matrix reuse
    already-built candidates rather than re-paying this multiplier each time.
    """

    method: Annotated[Literal["target_dim"], IDENTITY_ROLE] = "target_dim"
    target_coarse_dim: Annotated[int, Fact(TARGET_COARSE_DIMENSION)] = Field(
        gt=0, description="Desired realized coarse dimension."
    )
    theta_min: Annotated[float, OPERATIONAL_ROLE] = Field(
        default=0.01, gt=0.0, lt=1.0, description="Lower theta search bound."
    )
    theta_max: Annotated[float, OPERATIONAL_ROLE] = Field(
        default=0.99, gt=0.0, lt=1.0, description="Upper theta search bound."
    )
    step: Annotated[float, OPERATIONAL_ROLE] = Field(
        default=0.01, gt=0.0, description="Theta search grid spacing."
    )
    omega: Annotated[float | None, Fact(DAMPING, show=Show.NOT_NONE)] = Field(
        default=None,
        gt=0.0,
        description="Prolongation Jacobi-smoothing damping; auto per-matrix via "
        "torchalg's spectral-radius rule when unset.",
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    def exposed_checkpoint_ref(self) -> NeuralCheckpointRef | None:
        """Purely algebraic coarsening — never checkpoint-backed."""
        return None

    def slug(self) -> str:
        """Deterministic slug for this coarsening."""
        return (
            f"targetdim{self.target_coarse_dim}"
            f"_range{slug_number(self.theta_min)}to{slug_number(self.theta_max)}"
            f"_step{slug_number(self.step)}"
        )


class PODCoarseningConfig(BaseModel):
    """POD coarsening (the POD-2G of Nikolopoulos et al. 2022, §3.3-3.5).

    The prolongation/restriction operator is a POD basis fit to a snapshot
    ensemble of high-fidelity solutions (or, for a sharper coarse space,
    CG error traces e_k = x* - x_k, which are richer than raw solution vectors
    since e_0 == x* and later k emphasize slow-converging directions) —
    read from the train fold of an already-generated dataset directory, not raw
    files, so the same validation/normalization/manifest guarantees as every
    other dataset in this repo apply.

    The basis is fitted once per comparison method, before that method's
    per-matrix timed setups, and the fit's wall time and peak memory are
    measured and reported as the method's offline ``fit`` cost
    (``composition/comparison/_method_fit.py``). It is never checkpointed:
    the fit is cheap next to a comparison and its cost belongs to the method.

    Complexity: the fit is torchalg's ``compute_pod_basis`` over the ``m``
    snapshot rows of ``n`` DOFs, a thin SVD of ``O(min(m, n) * m * n)``, plus,
    for the energy inner product, the products with the sparse train-fold mean
    matrix ``W`` (``O(m * nnz(W))``). The per-matrix timed setup holds only
    the Galerkin coarse matrix ``Φᵀ(AΦ)`` in ``build_transfer``,
    ``O(rank * nnz(A) + rank² * n)`` on a CSR ``A``, and the coarse LU.
    Per-iteration application cost is ``O(rank * n)`` for the basis projection,
    next to the surrounding V-cycle's fine-level smoothing and residual cost —
    see ``AMGPreconditionerConfig``'s complexity note. See
    ``torchalg.preconditioners.implementations.pod`` and
    ``torchalg.utils.pod_basis`` for the derivation.
    """

    method: Annotated[Literal["pod"], IDENTITY_ROLE] = "pod"
    dataset_dir: Annotated[Path, InputData(), PROVENANCE_ROLE] = Field(
        ...,
        description="Generated dataset directory whose `solutions` array supplies POD snapshots.",
    )
    rank: Annotated[int, Fact(RANK)] = Field(
        default=8,
        ge=1,
        strict=True,
        description=(
            "Number of POD modes to retain: the coarse dimension. A rank that captures "
            "little of the snapshot energy is warned about at fit time."
        ),
    )
    n_snapshots: Annotated[SnapshotCount, Fact(SNAPSHOT_COUNT, show=Show.NON_DEFAULT)] = Field(
        default=ALL_SNAPSHOTS,
        description="Number of train-fold snapshot rows the fit reads; -1 means all.",
    )
    inner_product: Annotated[
        InnerProductKind, Fact(INNER_PRODUCT, convert=enum_value, show=Show.NON_DEFAULT)
    ] = Field(
        default=InnerProductKind.EUCLIDEAN,
        description=(
            "Inner product the basis is fitted in. Energy weighs it by the mean train-fold matrix."
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
            Anchored path string, or original value if not a string.
        """
        return expand_path_field(v, info)

    def inline_fit_rows(self, available: int) -> int:
        """How many of ``available`` train-fold snapshot rows the fit reads.

        The first ``n_snapshots`` of them, or all when ``n_snapshots`` is -1.
        """
        if self.n_snapshots == ALL_SNAPSHOTS:
            return available
        return min(self.n_snapshots, available)

    def exposed_checkpoint_ref(self) -> NeuralCheckpointRef | None:
        """Fitted inside the comparison — never checkpoint-backed."""
        return None

    def slug(self) -> str:
        """Deterministic slug for this coarsening, naming the dataset the basis is fitted on.

        The fit dataset distinguishes two POD methods of equal rank, as a checkpoint
        identity does for a fitted model; only its directory name enters, so a moved
        workspace keeps its keys.
        """
        base = f"pod_rank{slug_number(self.rank)}_n{self.n_snapshots}"
        if self.inner_product is InnerProductKind.ENERGY:
            base = f"{base}_energy"
        return f"{base}_data_{slug_text(self.dataset_dir.name)}"


class NeuralPODCoarseningConfig(NeuralCheckpointRef):
    """POD coarsening whose snapshot ensemble is predicted by a checkpoint.

    Same POD math as ``PODCoarseningConfig`` (the POD-2G of Nikolopoulos et al. 2022,
    §3.3-3.5), but the high-fidelity snapshot ensemble is *predicted* by a
    trained network instead of read from precomputed solutions: ``dataset_dir``
    supplies the network's input parameter arrays (not the solutions
    themselves), and the checkpoint identity fields inherited from
    ``NeuralCheckpointRef`` locate the model that turns those parameters into
    snapshots.

    Complexity: application cost is the same `O(rank * n)` basis projection
    as `PODCoarseningConfig` (also dominated in practice by the surrounding
    V-cycle's O(n^2) cost — same caveat as `PODCoarseningConfig`'s complexity
    note). Fit cost, unlike `PODCoarseningConfig`, is not
    a fixed SVD formula — the snapshot ensemble itself comes from a network
    forward pass (see `NeuralPreconditionerConfig`'s complexity note: model
    architecture-dependent, not stated here), *followed by* the same
    `O(min(m, n) * m * n)` SVD over the predicted snapshots.
    """

    method: Annotated[Literal["neural_pod"], IDENTITY_ROLE] = "neural_pod"
    dataset_dir: Annotated[Path, InputData(), PROVENANCE_ROLE] = Field(
        ...,
        description="Generated dataset directory whose `params` arrays feed the checkpoint.",
    )
    input_names: Annotated[tuple[str, ...], OPERATIONAL_ROLE] = Field(
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
    rank: Annotated[int, Fact(RANK)] = Field(
        default=8,
        ge=1,
        strict=True,
        description=(
            "Number of POD modes to retain: the coarse dimension. A rank that captures "
            "little of the snapshot energy is warned about at fit time."
        ),
    )
    n_snapshots: Annotated[SnapshotCount, Fact(SNAPSHOT_COUNT, show=Show.NON_DEFAULT)] = Field(
        default=ALL_SNAPSHOTS, description="Number of predicted snapshots to keep; -1 means all."
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
            Anchored path string, or original value if not a string.
        """
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

    def exposed_checkpoint_ref(self) -> NeuralCheckpointRef:
        """Always checkpoint-backed — its snapshot ensemble only exists via a checkpoint."""
        return self

    def slug(self) -> str:
        """Deterministic slug for this coarsening."""
        return (
            f"neuralpod_rank{slug_number(self.rank)}_n{self.n_snapshots}_{self.checkpoint_slug()}"
        )


CoarseningConfig = Annotated[
    AggregationCoarseningConfig
    | PODCoarseningConfig
    | NeuralPODCoarseningConfig
    | TargetDimCoarseningConfig,
    Field(discriminator="method"),
]
