"""Regression tests for generation strategy row-kind classification."""

from __future__ import annotations

import pytest

from neuralls.domain.generation.semantics import classify_strategy_row_kind
from neuralls.shared.types import GenerationStrategyKind


@pytest.mark.parametrize("kind", list(GenerationStrategyKind))
def test_every_strategy_kind_is_classified(kind: GenerationStrategyKind) -> None:
    """Every GenerationStrategyKind member must classify without raising."""
    classify_strategy_row_kind(kind)
