"""Validation tests for strict model_ref configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.preconditioner import (
    LoggedModelRefConfig,
    NeuralPreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
)


def test_registered_model_ref_requires_exactly_one_selector() -> None:
    """Registered refs must provide exactly one of alias/version/latest."""
    with pytest.raises(ValidationError):
        NeuralPreconditionerConfig(
            type=PreconditionerType.NEURAL,
            model_ref=RegisteredModelRefConfig(
                name="dataset-model",
                alias="candidate",
                version=1,
            ),
        )


def test_logged_model_ref_requires_filter_for_latest() -> None:
    """Logged latest lookup requires at least one filter."""
    with pytest.raises(ValidationError):
        NeuralPreconditionerConfig(
            type=PreconditionerType.NEURAL,
            model_ref=LoggedModelRefConfig(latest=True),
        )


def test_logged_model_ref_with_run_id_is_valid() -> None:
    """run_id selector is valid for logged model refs."""
    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(run_id="abc123"),
    )
    assert cfg.model_ref is not None
    assert cfg.model_ref.source == "run"


def test_assignment_and_registered_model_ref_name_together_is_rejected() -> None:
    """assignment and model_ref.name are competing identity sources; both is invalid."""
    with pytest.raises(ValidationError, match="not both"):
        NeuralPreconditionerConfig(
            type=PreconditionerType.NEURAL,
            assignment="my-assignment-id",
            model_ref=RegisteredModelRefConfig(name="unrelated-model", alias="champion"),
        )


def test_assignment_alone_with_registered_model_ref_alias_is_valid() -> None:
    """assignment plus an alias-only model_ref (no name) is the intended single-source case."""
    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        assignment="my-assignment-id",
        model_ref=RegisteredModelRefConfig(alias="champion"),
    )
    assert cfg.assignment == "my-assignment-id"
    assert isinstance(cfg.model_ref, RegisteredModelRefConfig)
    assert cfg.model_ref.name is None


def test_registered_model_ref_name_alone_without_assignment_is_valid() -> None:
    """model_ref.name with no assignment references an unrelated registered model."""
    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name="unrelated-model", alias="champion"),
    )
    assert cfg.assignment is None
    assert isinstance(cfg.model_ref, RegisteredModelRefConfig)
    assert cfg.model_ref.name == "unrelated-model"
