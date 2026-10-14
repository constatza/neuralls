"""The RHS mask is a sum type carried by every RHS source kind."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from neuralls.platform.config.models.comparison import (
    RHS_SOURCE_MODEL_BY_KIND,
    ComparisonRhsSourceModel,
    GaussianRhsSourceModel,
    NoMask,
    RandomMask,
)
from neuralls.shared.types import ComparisonRhsSourceKind

_SOURCE = TypeAdapter(ComparisonRhsSourceModel)

_REQUIRED_PARAMS: dict[ComparisonRhsSourceKind, dict[str, object]] = {
    ComparisonRhsSourceKind.GAUSSIAN: {},
    ComparisonRhsSourceKind.SINUSOIDAL: {"omega": 6.28},
    ComparisonRhsSourceKind.RAW_LHS: {"path": "/x"},
    ComparisonRhsSourceKind.RAW_RHS: {"path": "/x"},
    ComparisonRhsSourceKind.DATASET: {},
}


def test_the_mask_defaults_to_none() -> None:
    assert GaussianRhsSourceModel().mask == NoMask()


def test_the_density_defaults_to_a_tenth() -> None:
    assert RandomMask().density == 0.1


@pytest.mark.parametrize("density", [0.0, -0.1, 0.51, 1.0])
def test_a_density_outside_zero_to_a_half_is_rejected(density: float) -> None:
    with pytest.raises(ValidationError):
        RandomMask(density=density)


def test_a_density_of_exactly_a_half_is_accepted() -> None:
    assert RandomMask(density=0.5).density == 0.5


def test_every_kind_accepts_a_mask() -> None:
    assert set(_REQUIRED_PARAMS) == set(RHS_SOURCE_MODEL_BY_KIND)
    for kind, params in _REQUIRED_PARAMS.items():
        source = _SOURCE.validate_python(
            {"kind": kind, **params, "mask": {"kind": "random", "density": 0.2}}
        )

        assert source.mask == RandomMask(density=0.2), kind


def test_the_density_is_optional_in_toml_shape() -> None:
    source = _SOURCE.validate_python({"kind": "gaussian", "mask": {"kind": "random"}})

    assert source.mask == RandomMask()


def test_an_unknown_mask_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _SOURCE.validate_python({"kind": "gaussian", "mask": {"kind": "area"}})


def test_a_dumped_mask_validates_back_to_the_same_source() -> None:
    source = GaussianRhsSourceModel(mask=RandomMask(density=0.3))

    dumped = source.model_dump(exclude={"kind"})

    assert dumped["mask"] == {"kind": "random", "density": 0.3}
    assert GaussianRhsSourceModel.model_validate({"kind": "gaussian", **dumped}) == source
