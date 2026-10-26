"""Tests for cli/compare.py's outcome logging.

A comparison's reuse-check skip path returns ``ComparisonReused`` (no new
comparison result to report). This must never be logged as a failure.
"""

from __future__ import annotations

import pytest

from neuralls.composition.comparison.models import ComparisonOutcome


def test_comparison_outcome_rejects_success_with_an_error() -> None:
    """A successful comparison must not be representable with a failure reason."""
    with pytest.raises(TypeError):
        ComparisonOutcome(  # ty: ignore[call-non-callable]
            comparison_id="cmp-1",
            comparison_display_name="Comparison 1",
            success=True,
            error="contradictory state",
        )
