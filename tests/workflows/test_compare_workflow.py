"""Tests for the preconditioner result keys a comparison derives from its configs."""

from __future__ import annotations

import pytest

from neuralls.composition.comparison.result_keys import validate_unique_preconditioner_keys
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)


def test_preconditioner_result_keys_reject_duplicate_keys() -> None:
    """Two configs with identical type and fields derive the identical key and must be rejected.

    `name` is no longer user-settable — it's derived from type + fields (see
    `preconditioner.py::preconditioner_slug`), so a "duplicate key" now means
    what it should: two requested solves that are actually the same
    preconditioner, listed twice.
    """
    specs = (
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
    )

    with pytest.raises(ValueError):
        validate_unique_preconditioner_keys(specs)


def test_preconditioner_result_keys_differ_when_fields_differ() -> None:
    """Same type, different distinguishing field -> different derived keys, no collision."""
    specs = (
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI),
    )

    assert validate_unique_preconditioner_keys(specs) == ("identity", "jacobi")
