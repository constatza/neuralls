"""Tests for shared preconditioner scheduling config fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    IC0PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)


def test_start_iter_defaults_to_zero() -> None:
    """Preconditioner schedules activate immediately by default."""
    cfg = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    assert cfg.start_iter == 0


def test_start_iter_accepts_non_negative_value() -> None:
    """Preconditioner schedules accept delayed activation."""
    cfg = StandardPreconditionerConfig(
        type=PreconditionerType.JACOBI,
        start_iter=5,
    )
    assert cfg.start_iter == 5


def test_start_iter_rejects_negative_value() -> None:
    """Preconditioner schedules reject negative activation iterations."""
    with pytest.raises(ValidationError):
        StandardPreconditionerConfig(
            type=PreconditionerType.JACOBI,
            start_iter=-1,
        )


def test_name_is_derived_from_type_for_parameterless_variants() -> None:
    """A variant with no distinguishing fields derives `name` as just its `type`."""
    cfg = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    assert cfg.name == "jacobi"


def test_name_is_derived_from_distinguishing_fields() -> None:
    """`name` is never user-settable — it's always derived from type + fields."""
    assert IC0PreconditionerConfig().name == "ic0_threshold0"
    assert "aggregation_theta0p25_omegaauto" in (
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig()).name
    )


def test_name_ignores_a_legacy_name_field() -> None:
    """A stray `name=` key (old TOML files) is silently ignored, not stored or echoed back."""
    cfg = StandardPreconditionerConfig.model_validate({"name": "my-jacobi", "type": "jacobi"})
    assert cfg.name == "jacobi"


def test_name_collides_for_identical_configs_and_differs_otherwise() -> None:
    """Two configs with identical fields derive the identical name; a differing field changes it."""
    first = IC0PreconditionerConfig(threshold=0.001)
    second = IC0PreconditionerConfig(threshold=0.001)
    third = IC0PreconditionerConfig(threshold=0.002)
    assert first.name == second.name
    assert first.name != third.name
