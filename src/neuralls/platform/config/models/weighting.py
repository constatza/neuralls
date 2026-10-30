"""POD-2G snapshot-weighting configuration models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
)

from neuralls.platform.config.models.fact_roles import IDENTITY_ROLE
from neuralls.platform.config.models.slugs import slug_number
from neuralls.shared.fact_binding import (
    Fact,
    Show,
)
from neuralls.shared.scientific_description.catalog import (
    BETA,
    DAMPING,
    METRIC,
    STEPS,
    WEIGHTING,
)


class RawWeightingConfig(BaseModel):
    """No snapshot weighting — every snapshot votes equally in the POD SVD.

    The default `PODCoarseningConfig.weighting`, reproducing the exact
    behavior POD-2G had before per-snapshot weighting existed.
    """

    method: Annotated[Literal["raw"], IDENTITY_ROLE] = "raw"
    model_config = ConfigDict(extra="forbid", frozen=True)

    def slug(self) -> str:
        """Deterministic slug for this weighting."""
        return "raw"


class PowerNormWeightingConfig(BaseModel):
    """Row scale `||e_k||^(-beta)`, interpolating between raw and normalized snapshots.

    `beta=0` is `RawWeightingConfig` in disguise (every snapshot keeps its
    own magnitude); `beta=1` fully normalizes each snapshot by `metric`
    before the SVD — `"l2"` for `e_k / ||e_k||_2`, `"a"` for the A-inner-product
    `e_k / ||e_k||_A`. Values in between interpolate. See
    `torchalg.preconditioners.implementations.pod.weighting.power_norm_scales`,
    which this config resolves to.
    """

    method: Annotated[Literal["power_norm"], Fact(WEIGHTING)] = "power_norm"
    metric: Annotated[Literal["l2", "a"], Fact(METRIC)] = "l2"
    beta: Annotated[float, Fact(BETA)] = Field(
        default=1.0, ge=0.0, description="Normalization exponent."
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    def slug(self) -> str:
        """Deterministic slug for this weighting."""
        return f"powernorm_{self.metric}_beta{slug_number(self.beta)}"


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

    method: Annotated[Literal["smoother_persistence"], Fact(WEIGHTING)] = "smoother_persistence"
    steps: Annotated[int, Fact(STEPS)] = Field(
        default=5, gt=0, description="Number of damping sweeps."
    )
    omega: Annotated[float | None, Fact(DAMPING, show=Show.NOT_NONE)] = Field(
        default=None,
        gt=0.0,
        description="Weighted-Jacobi damping factor; auto per-matrix via "
        "torchalg's spectral-radius rule when unset.",
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    def slug(self) -> str:
        """Deterministic slug for this weighting."""
        omega_text = slug_number(self.omega) if self.omega is not None else "auto"
        return f"smoother_omega{omega_text}_steps{self.steps}"


SnapshotWeightingConfig = Annotated[
    RawWeightingConfig | PowerNormWeightingConfig | SmootherPersistenceWeightingConfig,
    Field(discriminator="method"),
]

parse_snapshot_weighting_config = TypeAdapter(SnapshotWeightingConfig).validate_python
"""Coerce a raw dict (as a TOML `[model]` table's `extra="allow"` passthrough
yields — see `composition/preconditioners/pod_fittable.py`) or an already-typed
instance into a validated `SnapshotWeightingConfig` member."""
