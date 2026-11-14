"""Placeholder residual measurements for tests that do not exercise them."""

from __future__ import annotations

from neuralls.domain.solver.models.benchmark_events import (
    ExactSolutionUnavailable,
    UnmeasuredResidual,
)
from neuralls.shared.device import SingleRun

SINGLE_RUN = SingleRun()
"""Repeat agreement of a solve timed once."""
NO_RESIDUAL = UnmeasuredResidual()
NO_REFERENCE = ExactSolutionUnavailable("not computed")
"""What ``compute_reference_solution`` returns when a test stubs the reference solve out."""
