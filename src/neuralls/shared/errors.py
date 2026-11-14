"""Errors shared by every pipeline stage."""

from __future__ import annotations

from enum import StrEnum
from typing import Self


class ReasonedError[Reason: StrEnum](Exception):
    """An error whose cause is a member of a closed enum, so callers match on it, not on text.

    ``detail`` is the human message. A cause that names absent items is its own subclass with
    a required ``missing`` field, so no error carries a field its reason leaves empty.
    ``__reduce__`` keeps the error picklable although ``__init__`` takes more than the message.

    Attributes:
        reason: The cause, one member of the subclass's reason enum.
        detail: The human-readable message.
    """

    def __init__(self, reason: Reason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail

    def __reduce__(self) -> tuple[type[Self], tuple[object, ...]]:
        """Rebuild from the constructor arguments, not the message alone."""
        return type(self), (self.reason, self.detail)


class StageInputsStaleReason(StrEnum):
    """Why an earlier stage's output cannot be consumed."""

    DATASET_STALE = "dataset_stale"
    NO_PROVENANCE = "no_provenance"
    SPLIT_NOT_CURRENT = "split_not_current"
    SPLIT_MISSING = "split_missing"


class StageInputsStale(ReasonedError[StageInputsStaleReason], RuntimeError):
    """An earlier stage's output is missing or no longer matches its hash.

    Raised by a stage that consumes another stage's artifacts (training reads
    the generated dataset and its pinned split; comparison additionally reads
    trained checkpoints). The consuming stage never produces those artifacts
    itself, so the only remedy is to run the named earlier stage.
    """
