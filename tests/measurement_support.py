"""Placeholder residual measurements for tests that do not exercise them."""

from __future__ import annotations

from neuralls.domain.solver.models.benchmark_events import (
    ExactSolutionUnavailable,
    UnmeasuredResidual,
)

NO_RESIDUAL = UnmeasuredResidual()
NO_REFERENCE = ExactSolutionUnavailable("not computed")
NO_REFERENCE_SOLUTION = (None, NO_REFERENCE)
"""What ``compute_reference_solution`` returns when a test stubs the reference solve out."""
