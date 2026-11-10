"""The stored form of a method's offline cost: one table per stage inside ``summary.toml``.

Each stage states its ``status`` and carries exactly the fields of that status, so a stage that
is unavailable stores no number and one that does not apply stores nothing else.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import assert_never

from neuralls.domain.solver.models.cost import (
    FitJob,
    FitOrigin,
    InlineFit,
    Known,
    NoFit,
    NotApplicable,
    OfflineStages,
    StageCost,
    Unavailable,
)
from neuralls.shared.types import CostProvenance

type StagePayload = dict[str, str | int | float]
type OfflinePayload = dict[str, StagePayload]


class StageStatus(StrEnum):
    """Which `StageCost` variant a stored stage is."""

    NOT_APPLICABLE = "not_applicable"
    KNOWN = "known"
    UNAVAILABLE = "unavailable"


def stage_status(cost: StageCost) -> StageStatus:
    """The status that names the variant of ``cost``."""
    match cost:
        case NotApplicable():
            return StageStatus.NOT_APPLICABLE
        case Known():
            return StageStatus.KNOWN
        case Unavailable():
            return StageStatus.UNAVAILABLE
        case unreachable:
            assert_never(unreachable)


class FitOriginKind(StrEnum):
    """Which `FitOrigin` variant a cost-breakdown row states."""

    NONE = "none"
    INLINE = "inline"
    JOB = "job"


def fit_origin_kind(origin: FitOrigin) -> FitOriginKind:
    """The kind that names the variant of ``origin``."""
    match origin:
        case NoFit():
            return FitOriginKind.NONE
        case InlineFit():
            return FitOriginKind.INLINE
        case FitJob():
            return FitOriginKind.JOB
        case unreachable:
            assert_never(unreachable)


def encode_offline_cost(offline: OfflineStages) -> OfflinePayload:
    """The ``[offline]`` tables of ``summary.toml`` for ``offline``."""
    return {
        "generation": _encode_stage(offline.generation),
        "training": _encode_stage(offline.training),
    }


def decode_offline_cost(payload: object) -> OfflineStages:
    """Read the ``[offline]`` tables back, rejecting a missing or malformed block."""
    if not isinstance(payload, dict):
        raise TypeError("the summary has no offline cost block")
    return OfflineStages(
        generation=_decode_stage(payload.get("generation"), "generation"),
        training=_decode_stage(payload.get("training"), "training"),
    )


def _encode_stage(cost: StageCost) -> StagePayload:
    status = stage_status(cost)
    match cost:
        case NotApplicable():
            return {"status": status.value}
        case Known() as known:
            stage: StagePayload = {
                "status": status.value,
                "seconds": known.wall_time_seconds,
                "provenance": known.provenance.value,
            }
            if known.peak_memory_bytes is not None:
                stage["peak_bytes"] = known.peak_memory_bytes
            return stage
        case Unavailable(reason=reason):
            return {"status": status.value, "reason": reason}
        case unreachable:
            assert_never(unreachable)


def _decode_stage(payload: object, name: str) -> StageCost:
    if not isinstance(payload, dict):
        raise TypeError(f"offline {name} stage is missing")
    match StageStatus(payload.get("status")):
        case StageStatus.NOT_APPLICABLE:
            return NotApplicable()
        case StageStatus.KNOWN:
            return Known(
                _non_negative_seconds(payload.get("seconds"), name),
                _peak_bytes(payload.get("peak_bytes"), name),
                CostProvenance(payload.get("provenance")),
            )
        case StageStatus.UNAVAILABLE:
            reason = payload.get("reason")
            if not isinstance(reason, str):
                raise TypeError(f"offline {name} stage has no reason")
            return Unavailable(reason)
        case unreachable:
            assert_never(unreachable)


def _non_negative_seconds(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"offline {name} seconds must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"offline {name} seconds must be finite and non-negative")
    return float(value)


def _peak_bytes(value: object, name: str) -> int | None:
    match value:
        case None:
            return None
        case int() if not isinstance(value, bool) and value >= 0:
            return value
        case _:
            raise ValueError(f"offline {name} peak bytes must be a non-negative integer")
