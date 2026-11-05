"""`OfflineStages.total` adds the stages that exist and never reads an unknown one as zero."""

from __future__ import annotations

import pytest

from neuralls.domain.solver.models.cost import (
    NO_OFFLINE_COST,
    Known,
    NotApplicable,
    OfflineStages,
    StageCost,
    Unavailable,
)
from neuralls.shared.types import CostProvenance

GENERATION = Known(10.0, 400, CostProvenance.HISTORICAL)
TRAINING = Known(90.0, 900, CostProvenance.HISTORICAL)


def test_total_sums_seconds_and_takes_the_largest_peak_memory() -> None:
    stages = OfflineStages(generation=GENERATION, training=TRAINING)

    assert stages.total() == Known(100.0, 900, CostProvenance.HISTORICAL)


def test_total_peak_memory_is_unknown_when_no_stage_recorded_one() -> None:
    stages = OfflineStages(
        generation=Known(1.0, None, CostProvenance.HISTORICAL),
        training=Known(2.0, None, CostProvenance.HISTORICAL),
    )

    assert stages.total() == Known(3.0, None, CostProvenance.HISTORICAL)


def test_total_keeps_the_peak_memory_of_the_only_stage_that_recorded_one() -> None:
    stages = OfflineStages(
        generation=Known(1.0, 700, CostProvenance.HISTORICAL),
        training=Known(2.0, None, CostProvenance.HISTORICAL),
    )

    assert stages.total() == Known(3.0, 700, CostProvenance.HISTORICAL)


def test_total_is_measured_only_when_every_stage_was_timed_live() -> None:
    live = Known(1.0, None, CostProvenance.MEASURED)

    assert OfflineStages(live, live).total() == Known(2.0, None, CostProvenance.MEASURED)
    assert OfflineStages(live, GENERATION).total() == Known(11.0, 400, CostProvenance.HISTORICAL)


@pytest.mark.parametrize("stage", [GENERATION, TRAINING])
def test_a_stage_that_does_not_apply_adds_nothing(stage: Known) -> None:
    assert OfflineStages(stage, NotApplicable()).total() == stage
    assert OfflineStages(NotApplicable(), stage).total() == stage


def test_no_stage_applying_totals_to_not_applicable() -> None:
    assert NO_OFFLINE_COST.total() == NotApplicable()


@pytest.mark.parametrize(
    "stages",
    [
        OfflineStages(GENERATION, Unavailable("no run id")),
        OfflineStages(Unavailable("no manifest"), TRAINING),
        OfflineStages(Unavailable("no manifest"), NotApplicable()),
        OfflineStages(NotApplicable(), Unavailable("no run id")),
    ],
)
def test_total_is_unavailable_when_any_stage_is(stages: OfflineStages) -> None:
    total = stages.total()

    assert isinstance(total, Unavailable)


def test_total_names_every_unavailable_stage_in_its_reason() -> None:
    stages = OfflineStages(Unavailable("no manifest"), Unavailable("no run id"))

    assert stages.total() == Unavailable("generation: no manifest; training: no run id")


def test_total_of_a_single_unavailable_stage_names_it() -> None:
    stages: StageCost = Unavailable("no run id")

    assert OfflineStages(GENERATION, stages).total() == Unavailable("training: no run id")
