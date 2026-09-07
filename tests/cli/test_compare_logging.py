"""Tests for cli/compare.py's outcome logging.

A comparison's reuse-check skip path returns success=True with payload=None
(no comparison result to report — the prior run's result was reused). This
must never be logged as a failure.
"""

from __future__ import annotations

from loguru import logger

from neuralls.cli.compare import _log_outcomes
from neuralls.composition.comparison.models import ComparisonOutcome


def test_cache_hit_outcome_is_not_logged_as_failed() -> None:
    """success=True with payload=None (the comparison reuse-check's skip path)
    must not fall into the 'Comparison failed' log branch."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        _log_outcomes(
            [
                ComparisonOutcome(
                    comparison_id="cmp-1",
                    comparison_display_name="Comparison 1",
                    success=True,
                    payload=None,
                )
            ]
        )
    finally:
        logger.remove(sink_id)

    assert not any("Comparison failed" in message for message in messages)
    assert any("existing MLflow comparison result" in message for message in messages)
