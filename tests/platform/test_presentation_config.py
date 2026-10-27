"""Tests for ``ScientificLabelsConfig`` (``docs/plan.md`` Phase 1, items 4-5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.presentation import ScientificLabelsConfig
from neuralls.shared.scientific_description.models import FieldKey
from neuralls.shared.scientific_description.policy import RenderTarget


def test_defaults_require_no_config() -> None:
    """An empty config still resolves a usable mathtext policy."""
    config = ScientificLabelsConfig()
    policy = config.notation_policy()
    assert policy.target is RenderTarget.MATHTEXT
    assert policy.symbol_for(FieldKey("iteration_count")).plain == "k"


def test_rendering_target_is_configurable() -> None:
    """``rendering`` selects plain/mathtext/latex."""
    config = ScientificLabelsConfig.model_validate({"rendering": "latex"})
    assert config.notation_policy().target is RenderTarget.LATEX


def test_rendering_rejects_unknown_target() -> None:
    """An unknown rendering target fails validation at the boundary."""
    with pytest.raises(ValidationError):
        ScientificLabelsConfig.model_validate({"rendering": "svg"})


def test_symbol_override_from_toml_shape() -> None:
    """A ``[scientific_labels.symbols.sample_count]`` table overrides one symbol."""
    config = ScientificLabelsConfig.model_validate(
        {
            "rendering": "plain",
            "symbols": {"sample_count": {"plain": "N_train", "math": "N_{\\mathrm{train}}"}},
        }
    )
    symbol = config.notation_policy().symbol_for(FieldKey("sample_count"))
    assert symbol.plain == "N_train"
    assert symbol.math == "N_{\\mathrm{train}}"


def test_symbol_override_does_not_disturb_other_defaults() -> None:
    """Overriding one symbol leaves the rest of the default table intact."""
    config = ScientificLabelsConfig.model_validate(
        {"symbols": {"sample_count": {"plain": "N_train", "math": "N_{\\mathrm{train}}"}}}
    )
    policy = config.notation_policy()
    assert policy.symbol_for(FieldKey("iteration_count")).plain == "k"


def test_unknown_field_rejected() -> None:
    """Extra top-level fields fail validation (``extra=\"forbid\"``)."""
    with pytest.raises(ValidationError):
        ScientificLabelsConfig.model_validate({"unexpected": 1})
