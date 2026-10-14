"""Errors shared by every pipeline stage."""

from __future__ import annotations


class StageInputsStale(RuntimeError):
    """An earlier stage's output is missing or no longer matches its hash.

    Raised by a stage that consumes another stage's artifacts (training reads
    the generated dataset and its pinned split; comparison additionally reads
    trained checkpoints). The consuming stage never produces those artifacts
    itself, so the only remedy is to run the named earlier stage.
    """
