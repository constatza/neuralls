"""`OfflineStages.combined` adds the stages that exist and never reads an unknown one as zero."""

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
FIT = Known(90.0, 900, CostProvenance.HISTORICAL)


def test_combined_sums_seconds_and_takes_the_largest_peak_memory() -> None:
    stages = OfflineStages(generation=GENERATION, fit=FIT)

    assert stages.combined() == Known(100.0, 900, CostProvenance.HISTORICAL)


def test_combined_peak_memory_is_unknown_when_no_stage_recorded_one() -> None:
    stages = OfflineStages(
        generation=TrainShareOfGeneration(Known(1.0, None, CostProvenance.HISTORICAL), WHOLE),
        fit=Known(2.0, None, CostProvenance.HISTORICAL),
    )

    assert stages.combined() == Known(3.0, None, CostProvenance.HISTORICAL)


def test_combined_keeps_the_peak_memory_of_the_only_stage_that_recorded_one() -> None:
    stages = OfflineStages(
        generation=TrainShareOfGeneration(Known(1.0, 700, CostProvenance.HISTORICAL), WHOLE),
        fit=Known(2.0, None, CostProvenance.HISTORICAL),
    )

    assert stages.combined() == Known(3.0, 700, CostProvenance.HISTORICAL)


def test_combined_is_measured_only_when_every_stage_was_timed_live() -> None:
    live = Known(1.0, None, CostProvenance.MEASURED)

    live_generation = TrainShareOfGeneration(live, WHOLE)

    assert OfflineStages(live_generation, live).combined() == Known(
        2.0, None, CostProvenance.MEASURED
    )
    assert OfflineStages(live_generation, FIT).combined() == Known(
        91.0, 900, CostProvenance.HISTORICAL
    )


def test_a_stage_that_does_not_apply_adds_nothing() -> None:
    assert OfflineStages(GENERATION, NotApplicable()).combined() == GENERATED
    assert OfflineStages(NotApplicable(), FIT).combined() == FIT


def test_no_stage_applying_combine_to_not_applicable() -> None:
    assert NO_OFFLINE_COST.combined() == NotApplicable()


@pytest.mark.parametrize(
    ("stages", "reason"),
    [
        (OfflineStages(GENERATION, Unavailable("no run id")), "fit: no run id"),
        (OfflineStages(Unavailable("no manifest"), FIT), "generation: no manifest"),
        (OfflineStages(Unavailable("no manifest"), NotApplicable()), "generation: no manifest"),
        (OfflineStages(NotApplicable(), Unavailable("no run id")), "fit: no run id"),
        (
            OfflineStages(Unavailable("no manifest"), Unavailable("no run id")),
            "generation: no manifest; fit: no run id",
        ),
    ],
    ids=[
        "fit_unavailable",
        "generation_unavailable",
        "generation_unavailable_fit_not_applicable",
        "fit_unavailable_generation_not_applicable",
        "both_unavailable",
    ],
)
def test_combined_is_unavailable_and_names_every_unavailable_stage(
    stages: OfflineStages, reason: str
) -> None:
    assert stages.combined() == Unavailable(reason)


UNAVAILABLE = Unavailable("not recorded")
MEASURED_FIT = Known(3.0, 100, CostProvenance.MEASURED)


@pytest.mark.parametrize(
    ("generation", "fit", "expected"),
    [
        (NotApplicable(), NotApplicable(), NoFit()),
        (GENERATION, MEASURED_FIT, InlineFit()),
        (UNAVAILABLE, MEASURED_FIT, InlineFit()),
        (GENERATION, FIT, FitJob()),
        (UNAVAILABLE, FIT, FitJob()),
        (GENERATION, UNAVAILABLE, FitJob()),
        (NotApplicable(), UNAVAILABLE, FitJob()),
    ],
)
def test_the_fit_origin_is_read_from_the_fit_stage_provenance(
    generation: GenerationCost, fit: StageCost, expected: FitOrigin
) -> None:
    assert OfflineStages(generation=generation, fit=fit).fit_origin() == expected
