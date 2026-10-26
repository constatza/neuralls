"""Tests for cli/compare.py's outcome logging.

A comparison's reuse-check skip path returns ``ComparisonReused`` (no new
comparison result to report). This must never be logged as a failure.
"""

from __future__ import annotations

import pytest
from loguru import logger

from neuralls.cli.compare import _log_outcomes
from neuralls.composition.comparison.models import ComparisonOutcome, ComparisonReused


def test_comparison_outcome_rejects_success_with_an_error() -> None:
    """A successful comparison must not be representable with a failure reason."""
    with pytest.raises(TypeError):
        ComparisonOutcome(  # ty: ignore[call-non-callable]
            comparison_id="cmp-1",
            comparison_display_name="Comparison 1",
            success=True,
            error="contradictory state",
        )


def test_cache_hit_outcome_is_not_logged_as_failed() -> None:
    """The explicit reuse state must not fall into the failure log branch."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        _log_outcomes(
            [
                ComparisonReused(
                    comparison_id="cmp-1",
                    comparison_display_name="Comparison 1",
                )
            ]
        )
    finally:
        logger.remove(sink_id)

    assert not any("Comparison failed" in message for message in messages)
    assert any("existing MLflow comparison result" in message for message in messages)
