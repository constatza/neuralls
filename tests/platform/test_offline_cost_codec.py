"""The offline cost is stored as one table per stage and read back as the same typed stages."""

from __future__ import annotations

import math

import pytest

from neuralls.domain.solver.models.cost import (
    NO_OFFLINE_COST,
    Known,
    NotApplicable,
    OfflineStages,
    Unavailable,
)
from neuralls.platform.reporting.offline_cost_codec import (
    decode_offline_cost,
    encode_offline_cost,
)
from neuralls.shared.types import CostProvenance

OFFLINE_COSTS = (
    NO_OFFLINE_COST,
    OfflineStages(Known(8.5, 1024, CostProvenance.HISTORICAL), NotApplicable()),
    OfflineStages(
        Known(8.5, None, CostProvenance.HISTORICAL), Known(90.25, None, CostProvenance.HISTORICAL)
    ),
    OfflineStages(Known(0.0, 7, CostProvenance.MEASURED), Unavailable("no recorded training run")),
    OfflineStages(Unavailable("no manifest"), Unavailable("run gone cannot be read")),
)


@pytest.mark.parametrize("offline", OFFLINE_COSTS)
def test_an_offline_cost_is_read_back_as_written(offline: OfflineStages) -> None:
    assert decode_offline_cost(encode_offline_cost(offline)) == offline


def test_a_stage_is_stored_with_its_status_and_only_the_fields_of_that_status() -> None:
    offline = OfflineStages(
        Known(8.5, 1024, CostProvenance.HISTORICAL), Unavailable("no recorded training run")
    )

    assert encode_offline_cost(offline) == {
        "generation": {
            "status": "known",
            "seconds": 8.5,
            "provenance": "historical",
            "peak_bytes": 1024,
        },
        "training": {"status": "unavailable", "reason": "no recorded training run"},
    }


def test_a_stage_that_does_not_apply_stores_only_its_status() -> None:
    assert encode_offline_cost(NO_OFFLINE_COST) == {
        "generation": {"status": "not_applicable"},
        "training": {"status": "not_applicable"},
    }


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"generation": {"status": "not_applicable"}},
        {"generation": {"status": "not_applicable"}, "training": "known"},
        {"generation": {"status": "free"}, "training": {"status": "not_applicable"}},
        {
            "generation": {"status": "known", "seconds": 1.0, "provenance": "guessed"},
            "training": {"status": "not_applicable"},
        },
        {
            "generation": {"status": "known", "provenance": "historical"},
            "training": {"status": "not_applicable"},
        },
        {
            "generation": {"status": "known", "seconds": -1.0, "provenance": "historical"},
            "training": {"status": "not_applicable"},
        },
        {
            "generation": {"status": "known", "seconds": math.nan, "provenance": "historical"},
            "training": {"status": "not_applicable"},
        },
        {
            "generation": {
                "status": "known",
                "seconds": 1.0,
                "provenance": "historical",
                "peak_bytes": -3,
            },
            "training": {"status": "not_applicable"},
        },
        {
            "generation": {"status": "unavailable"},
            "training": {"status": "not_applicable"},
        },
    ],
)
def test_a_malformed_or_missing_offline_block_is_rejected(payload: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        decode_offline_cost(payload)
