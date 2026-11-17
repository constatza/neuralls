"""The POD recipe: a ``run.type = "pod"`` job that the comparison fits, not dlkit.

A POD basis needs no trainer, seed, precision or data profile: an assignment pairs the
recipe with a dataset, and the comparison fits the basis once per method from that
dataset's pinned train fold, before the timed setups. So the recipe carries only what
the fit reads.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt

from neuralls.platform.config.models.coarsening import ALL_SNAPSHOTS, SnapshotCount
from neuralls.shared.types import InnerProductKind

POD_RUN_TYPE = "pod"
"""The ``run.type`` that marks a job TOML as a POD recipe."""


class PODRunSection(BaseModel):
    """The ``[run]`` table of a POD recipe: its kind and nothing else."""

    type: Literal["pod"]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PODSection(BaseModel):
    """The ``[pod]`` table: the fit's mode count, inner product and snapshot cut."""

    rank: PositiveInt
    inner_product: InnerProductKind = InnerProductKind.EUCLIDEAN
    n_snapshots: SnapshotCount = ALL_SNAPSHOTS
    model_config = ConfigDict(extra="forbid", frozen=True)


class PODJobConfig(BaseModel):
    """One POD recipe job TOML (``[run] type = "pod"`` plus ``[pod]``)."""

    run: PODRunSection
    pod: PODSection
    model_config = ConfigDict(extra="forbid", frozen=True)
