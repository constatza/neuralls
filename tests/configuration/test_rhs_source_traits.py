"""RHS source models: the kind set is closed, and each kind declares its own traits."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from neuralls.composition.comparison._linear_system import _rhs_is_generated
from neuralls.platform.config.models.comparison import (
    RHS_SOURCE_MODEL_BY_KIND,
    ComparisonRhsSourceModel,
)
from neuralls.shared.types import ComparisonRhsSourceKind

_SOURCE = TypeAdapter(ComparisonRhsSourceModel)


@pytest.mark.parametrize("kind", ["cross_dataset", "random_sparse"])
def test_removed_kinds_are_rejected_as_unknown(kind: str) -> None:
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        _SOURCE.validate_python({"kind": kind, "path": "/data/other"})


def test_every_kind_has_exactly_one_source_model() -> None:
    assert set(RHS_SOURCE_MODEL_BY_KIND) == set(ComparisonRhsSourceKind)


@pytest.mark.parametrize(
    ("kind", "generated", "seeded", "single_draw"),
    [
        (ComparisonRhsSourceKind.GAUSSIAN, True, True, False),
        (ComparisonRhsSourceKind.SINUSOIDAL, True, True, False),
        (ComparisonRhsSourceKind.RAW_LHS, False, False, True),
        (ComparisonRhsSourceKind.RAW_RHS, False, False, True),
        (ComparisonRhsSourceKind.DATASET, False, False, False),
    ],
)
def test_each_kind_declares_its_traits(
    kind: ComparisonRhsSourceKind, generated: bool, seeded: bool, single_draw: bool
) -> None:
    model = RHS_SOURCE_MODEL_BY_KIND[kind]

    assert (model.generated, model.seeded, model.single_draw) == (generated, seeded, single_draw)


def test_normalization_treats_exactly_the_generated_kinds_as_independently_scaled() -> None:
    assert {kind for kind in ComparisonRhsSourceKind if _rhs_is_generated(kind)} == {
        ComparisonRhsSourceKind.GAUSSIAN,
        ComparisonRhsSourceKind.SINUSOIDAL,
    }
    assert _rhs_is_generated(None) is False
