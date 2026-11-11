"""An explicit shuffle request is warned about; the default stays silent."""

from __future__ import annotations

from neuralls.composition.generation._context_builder import _warn_if_shuffle_requested
from neuralls.platform.config.models.data_models import GenerationConfig


def test_explicit_shuffle_true_logs_one_warning(warning_messages: list[str]) -> None:
    _warn_if_shuffle_requested(GenerationConfig.model_validate({"shuffle": True}))
    assert len(warning_messages) == 1


def test_default_shuffle_does_not_warn(warning_messages: list[str]) -> None:
    _warn_if_shuffle_requested(GenerationConfig())
    assert warning_messages == []
