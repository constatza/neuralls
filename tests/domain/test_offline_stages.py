"""`OfflineStages.total` adds the stages that exist and never reads an unknown one as zero."""

from __future__ import annotations

import pytest

from neuralls.domain.solver.models.cost import (
    NO_OFFLINE_COST,
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

WHOLE = TrainShare(1.0, FitFolds.TRAIN)
GENERATED = Known(10.0, 400, CostProvenance.HISTORICAL)
GENERATION = TrainShareOfGeneration(GENERATED, WHOLE)
TRAINING = Known(90.0, 900, CostProvenance.HISTORICAL)


def test_total_sums_seconds_and_takes_the_largest_peak_memory() -> None:
    stages = OfflineStages(generation=GENERATION, training=TRAINING)

    assert stages.total() == Known(100.0, 900, CostProvenance.HISTORICAL)


def test_total_peak_memory_is_unknown_when_no_stage_recorded_one() -> None:
    stages = OfflineStages(
        generation=TrainShareOfGeneration(Known(1.0, None, CostProvenance.HISTORICAL), WHOLE),
        training=Known(2.0, None, CostProvenance.HISTORICAL),
    )

    assert stages.total() == Known(3.0, None, CostProvenance.HISTORICAL)


def test_total_keeps_the_peak_memory_of_the_only_stage_that_recorded_one() -> None:
    stages = OfflineStages(
        generation=TrainShareOfGeneration(Known(1.0, 700, CostProvenance.HISTORICAL), WHOLE),
        training=Known(2.0, None, CostProvenance.HISTORICAL),
    )

    assert stages.total() == Known(3.0, 700, CostProvenance.HISTORICAL)


def test_total_is_measured_only_when_every_stage_was_timed_live() -> None:
    live = Known(1.0, None, CostProvenance.MEASURED)

    live_generation = TrainShareOfGeneration(live, WHOLE)

    assert OfflineStages(live_generation, live).total() == Known(2.0, None, CostProvenance.MEASURED)
    assert OfflineStages(live_generation, TRAINING).total() == Known(
        91.0, 900, CostProvenance.HISTORICAL
    )


def test_a_stage_that_does_not_apply_adds_nothing() -> None:
    assert OfflineStages(GENERATION, NotApplicable()).total() == GENERATED
    assert OfflineStages(NotApplicable(), TRAINING).total() == TRAINING


def test_no_stage_applying_totals_to_not_applicable() -> None:
    assert NO_OFFLINE_COST.total() == NotApplicable()


@pytest.mark.parametrize(
    ("stages", "reason"),
    [
        (OfflineStages(GENERATION, Unavailable("no run id")), "training: no run id"),
        (OfflineStages(Unavailable("no manifest"), TRAINING), "generation: no manifest"),
        (OfflineStages(Unavailable("no manifest"), NotApplicable()), "generation: no manifest"),
        (OfflineStages(NotApplicable(), Unavailable("no run id")), "training: no run id"),
        (
            OfflineStages(Unavailable("no manifest"), Unavailable("no run id")),
            "generation: no manifest; training: no run id",
        ),
    ],
    ids=[
        "training_unavailable",
        "generation_unavailable",
        "generation_unavailable_training_not_applicable",
        "training_unavailable_generation_not_applicable",
        "both_unavailable",
    ],
)
def test_total_is_unavailable_and_names_every_unavailable_stage(
    stages: OfflineStages, reason: str
) -> None:
    assert stages.total() == Unavailable(reason)


UNAVAILABLE = Unavailable("not recorded")


@pytest.mark.parametrize(
    ("generation", "training", "expected"),
    [
        (NotApplicable(), NotApplicable(), NoFit()),
        (GENERATION, NotApplicable(), InlineFit()),
        (UNAVAILABLE, NotApplicable(), InlineFit()),
        (NotApplicable(), TRAINING, FitJob()),
        (NotApplicable(), UNAVAILABLE, FitJob()),
        (GENERATION, TRAINING, FitJob()),
        (GENERATION, UNAVAILABLE, FitJob()),
        (UNAVAILABLE, TRAINING, FitJob()),
        (UNAVAILABLE, UNAVAILABLE, FitJob()),
    ],
)
def test_the_fit_origin_is_read_from_which_stages_apply(
    generation: GenerationCost, training: StageCost, expected: FitOrigin
) -> None:
    assert OfflineStages(generation=generation, training=training).fit_origin() == expected
