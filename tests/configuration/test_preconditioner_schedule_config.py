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
    cfg = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    assert cfg.start_iter == 0


def test_start_iter_accepts_non_negative_value() -> None:
    """Preconditioner schedules accept delayed activation."""
    cfg = StandardPreconditionerConfig(
        name="jacobi",
        type=PreconditionerType.JACOBI,
        start_iter=5,
    )
    assert cfg.start_iter == 5


def test_start_iter_rejects_negative_value() -> None:
    """Preconditioner schedules reject negative activation iterations."""
    with pytest.raises(ValidationError):
        StandardPreconditionerConfig(
            name="jacobi",
            type=PreconditionerType.JACOBI,
            start_iter=-1,
        )


def test_name_defaults_to_type_when_omitted() -> None:
    """Omitting `name` falls back to the concrete subclass's `type` value."""
    cfg = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    assert cfg.name == "jacobi"


def test_name_default_falls_back_per_subclass_type() -> None:
    """Each preconditioner subclass defaults `name` to its own `type`, not a shared constant."""
    assert IC0PreconditionerConfig().name == "ic0"
    assert AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig()).name == "amg"


def test_explicit_name_is_not_overridden_by_type() -> None:
    """An explicitly given `name` is kept as-is, even though it differs from `type`."""
    cfg = StandardPreconditionerConfig(name="my-jacobi", type=PreconditionerType.JACOBI)
    assert cfg.name == "my-jacobi"
