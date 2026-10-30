"""AMG coarsening configuration models (algebraic and POD-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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
from neuralls.platform.config.models.slugs import slug_number
from neuralls.platform.config.models.weighting import RawWeightingConfig, SnapshotWeightingConfig
from neuralls.shared.digest import InputData
from neuralls.shared.fact_binding import (
    Fact,
    Nested,
    Show,
)
from neuralls.shared.scientific_description.catalog import (
    DAMPING,
    RANK,
    SNAPSHOT_COUNT,
    TARGET_COARSE_DIMENSION,
    THRESHOLD,
)


def _validate_pod_rank(v: float) -> int | float:
    """Enforce the per-mode constraint matching whichever branch of ``rank`` was given.

    Shared by ``PODCoarseningConfig._validate_rank`` and
    ``NeuralPODCoarseningConfig._validate_rank`` — same POD-2G rank
    contract, only the surrounding config differs.

    Args:
        v: The parsed ``rank`` value — an int (mode count) or float
            (energy threshold).

    Returns:
        The validated value, unchanged.

    Raises:
        ValueError: If an int rank is < 1, or a float rank is outside (0, 1].
    """
    if isinstance(v, float) and not (0.0 < v <= 1.0):
        raise ValueError(f"rank as an energy threshold must be in (0, 1], got {v}")
    if not isinstance(v, float) and v < 1:
        raise ValueError(f"rank as a mode count must be >= 1, got {v}")
    return v


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

    Complexity: fit cost has two parts — the thin SVD over the snapshot
    matrix (`m` snapshots x `n` DOFs), `O(min(m, n) * m * n)` for
    `torch.linalg.svd`'s default full-matrices=False mode (cheap when
    `m << n`, the usual case), and the one-time Galerkin coarse-matrix
    triple product `Phi_r^T @ A @ Phi_r` in `build_transfer`, `O(rank * n^2)`
    (dense `n x n` matrix, dominates the SVD when `rank` isn't tiny relative
    to `m`). No fit cost at all when checkpoint-backed (see the class
    docstring above) — just a checkpoint load (done ahead of the timed build, once
    per dataset), though `build_transfer`'s
    triple product still runs once per comparison since it depends on the
    matrix, not the basis. Per-iteration application cost is `O(rank * n)`
    for the basis projection itself, but every CG iteration through
    `AMGPreconditioner`'s V-cycle also pays the surrounding cycle's O(n^2)
    dense fine-level smoothing/residual cost — see
    `AMGPreconditionerConfig`'s complexity note; POD-2G's basis projection
    is cheap next to that, not instead of it. See
    `torchalg.preconditioners.implementations.pod.coarsening` /
    `.basis` for the derivation.
    """

    method: Annotated[Literal["pod"], IDENTITY_ROLE] = "pod"
    dataset_dir: Annotated[Path, InputData(), PROVENANCE_ROLE] = Field(
        ...,
        description="Generated dataset directory whose `solutions` array supplies POD-2G snapshots.",
    )
    rank: Annotated[int | float, Fact(RANK)] = Field(
        default=8,
        description=(
            "Fixed number of POD modes to retain (int), or minimum cumulative "
            "captured energy to retain (float in (0, 1] — e.g. 0.9999)."
        ),
    )
    n_snapshots: Annotated[int, Fact(SNAPSHOT_COUNT, show=Show.NON_DEFAULT)] = Field(
        default=-1, description="Number of snapshot files to load; -1 means all matched."
    )
    weighting: Annotated[SnapshotWeightingConfig, Nested()] = Field(
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
        return _validate_pod_rank(v)

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

    def slug(self) -> str:
        """Deterministic slug for this coarsening, including any checkpoint identity."""
        base = f"pod_rank{slug_number(self.rank)}_n{self.n_snapshots}_{self.weighting.slug()}"
        ref = self.exposed_checkpoint_ref()
        return f"{base}_{ref.checkpoint_slug()}" if ref is not None else base


class NeuralPODCoarseningConfig(NeuralCheckpointRef):
    """POD-2G coarsening whose snapshot ensemble is predicted by a checkpoint.

    Same POD-2G math as ``PODCoarseningConfig`` (Nikolopoulos et al. 2022,
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
    rank: Annotated[int | float, Fact(RANK)] = Field(
        default=8,
        description=(
            "Fixed number of POD modes to retain (int), or minimum cumulative "
            "captured energy to retain (float in (0, 1] — e.g. 0.9999)."
        ),
    )
    n_snapshots: Annotated[int, Fact(SNAPSHOT_COUNT, show=Show.NON_DEFAULT)] = Field(
        default=-1, description="Number of predicted snapshots to keep; -1 means all."
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
        return _validate_pod_rank(v)

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
