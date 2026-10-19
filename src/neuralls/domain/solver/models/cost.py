"""One pipeline stage's wall-clock cost and where the number came from."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.shared.types import CostProvenance


@dataclass(frozen=True, slots=True)
class StageCost:
    """One pipeline stage's wall-clock cost, tagged with where the number came from.

    See shared.types.CostProvenance for what MEASURED/HISTORICAL/UNAVAILABLE mean.
    """

    wall_time_seconds: float
    peak_memory_bytes: int | None
    provenance: CostProvenance
