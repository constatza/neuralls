"""The stored form of a method's offline cost: one table per stage inside ``summary.toml``.

Each stage states its ``status`` and carries exactly the fields of that status, so a stage that
is unavailable stores no number and one that does not apply stores nothing else. A known
generation is the whole dataset's ``dataset_seconds``, the ``train_share`` of them charged and the
``fit_folds`` that share covers. A summary without the folds is rejected, so a method stored before
they were recorded is run again rather than read with a share of unknown rows.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import assert_never

from neuralls.domain.solver.models.cost import (
    FitFolds,
    FitJob,
    FitOrigin,
    GenerationCost,
    InlineFit,
    Known,
    NoFit,
    NotApplicable,
    OfflineStages,
    StageCost,
    TrainShare,
    TrainShareOfGeneration,
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
        "generation": _encode_generation(offline.generation),
        "training": _encode_stage(offline.training),
    }


def decode_offline_cost(payload: object) -> OfflineStages:
    """Read the ``[offline]`` tables back, rejecting a missing or malformed block."""
    if not isinstance(payload, dict):
        raise TypeError("the summary has no offline cost block")
    return OfflineStages(
        generation=_decode_generation(payload.get("generation")),
        training=_decode_stage(payload.get("training"), "training"),
    )


def _encode_generation(cost: GenerationCost) -> StagePayload:
    match cost:
        case TrainShareOfGeneration(dataset=dataset, share=share):
            stage: StagePayload = {
                "status": StageStatus.KNOWN.value,
                "dataset_seconds": dataset.wall_time_seconds,
                "train_share": share.value,
                "fit_folds": share.folds.value,
                "provenance": dataset.provenance.value,
            }
            if dataset.peak_memory_bytes is not None:
                stage["peak_bytes"] = dataset.peak_memory_bytes
            return stage
        case NotApplicable() | Unavailable():
            return _encode_stage(cost)
        case unreachable:
            assert_never(unreachable)


def _decode_generation(payload: object) -> GenerationCost:
    if not isinstance(payload, dict):
        raise TypeError("offline generation stage is missing")
    match StageStatus(payload.get("status")):
        case StageStatus.NOT_APPLICABLE:
            return NotApplicable()
        case StageStatus.KNOWN:
            return TrainShareOfGeneration(
                _decode_known(payload, "dataset_seconds", "generation"),
                TrainShare(
                    _number(payload.get("train_share"), "offline generation train share"),
                    FitFolds(payload.get("fit_folds")),
                ),
            )
        case StageStatus.UNAVAILABLE:
            return _decode_unavailable(payload, "generation")
        case unreachable:
            assert_never(unreachable)


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
            return _decode_known(payload, "seconds", name)
        case StageStatus.UNAVAILABLE:
            return _decode_unavailable(payload, name)
        case unreachable:
            assert_never(unreachable)


def _decode_unavailable(payload: dict[str, object], name: str) -> Unavailable:
    reason = payload.get("reason")
    if not isinstance(reason, str):
        raise TypeError(f"offline {name} stage has no reason")
    return Unavailable(reason)


def _decode_known(payload: dict[str, object], seconds_key: str, name: str) -> Known:
    return Known(
        _non_negative_seconds(payload.get(seconds_key), name),
        _peak_bytes(payload.get("peak_bytes"), name),
        CostProvenance(payload.get("provenance")),
    )


def _number(value: object, what: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{what} must be a number")
    return float(value)


def _non_negative_seconds(value: object, name: str) -> float:
    value = _number(value, f"offline {name} seconds")
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
