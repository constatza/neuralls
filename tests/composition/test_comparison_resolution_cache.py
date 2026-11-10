"""A batch resolves the checkpoints of identical preconditioner sets once."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuralls.composition.assignments.comparison_resolution import (
    ComparisonResolutionContext,
    resolve_comparison_specs,
)
from neuralls.platform.config.models.checkpoint_refs import ByAlias, RegisteredModelRefConfig
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
)
from tests.presentation_support import NO_DATASET_READS
from tests.workflows.comparison_support import mock_cfg

_RESOLVE_MODELS = "neuralls.composition.assignments.comparison_resolution.resolve_preconditioner_models_with_warnings"


@pytest.fixture
def neural_spec() -> NeuralPreconditionerConfig:
    return NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        architecture="Linear Net",
        model_ref=RegisteredModelRefConfig(name="net", selector=ByAlias(alias="best")),
    )


@pytest.fixture
def context() -> ComparisonResolutionContext:
    return ComparisonResolutionContext(
        topology=MagicMock(),
        case_config_path=Path("case.toml"),
        settings=MagicMock(),
        load_dataset=NO_DATASET_READS,
        tracking=MagicMock(),
    )


def test_entries_with_the_same_preconditioners_resolve_once(
    context: ComparisonResolutionContext, neural_spec: NeuralPreconditionerConfig
) -> None:
    first = mock_cfg(preconditioners=[neural_spec])
    second = mock_cfg(preconditioners=[neural_spec])
    with patch(_RESOLVE_MODELS) as resolve:
        resolve.return_value = MagicMock(specs=[neural_spec], warnings=())
        resolved_first = resolve_comparison_specs(first, context, MagicMock())
        resolved_second = resolve_comparison_specs(second, context, MagicMock())

    assert resolve.call_count == 1
    assert resolved_second.specs == resolved_first.specs
    assert resolved_second.checkpoint_digests == resolved_first.checkpoint_digests


def test_a_different_dataset_alias_resolves_separately(
    context: ComparisonResolutionContext, neural_spec: NeuralPreconditionerConfig
) -> None:
    first = mock_cfg(preconditioners=[neural_spec])
    second = mock_cfg(preconditioners=[neural_spec])
    second.general.data.dataset_alias = "other"
    with patch(_RESOLVE_MODELS) as resolve:
        resolve.return_value = MagicMock(specs=[neural_spec], warnings=())
        resolve_comparison_specs(first, context, MagicMock())
        resolve_comparison_specs(second, context, MagicMock())

    assert resolve.call_count == 2


def test_a_different_preconditioner_set_resolves_separately(
    context: ComparisonResolutionContext, neural_spec: NeuralPreconditionerConfig
) -> None:
    other = neural_spec.model_copy(
        update={"model_ref": RegisteredModelRefConfig(name="net", selector=ByAlias(alias="old"))}
    )
    with patch(_RESOLVE_MODELS) as resolve:
        resolve.return_value = MagicMock(specs=[neural_spec], warnings=())
        resolve_comparison_specs(mock_cfg(preconditioners=[neural_spec]), context, MagicMock())
        resolve_comparison_specs(mock_cfg(preconditioners=[other]), context, MagicMock())

    assert resolve.call_count == 2
