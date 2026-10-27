"""Text the command line prints about results, rendered from their typed presentations.

Every label here is the plain text of the same presentation MLflow names its runs with, so a
line printed by a command and the run it refers to read alike. Nothing is built from an id.
"""

from __future__ import annotations

from neuralls.application.models import AssignmentResult
from neuralls.composition.comparison.models import UndescribedComparison
from neuralls.composition.generation.multi_generation import GenerationResult
from neuralls.domain.presentation.run_names import NamedRun
from neuralls.platform.tracking.run_names import (
    render_dataset_label,
    render_training_label,
    run_name_text,
)

UNDESCRIBED_COMPARISON_TEXT = "Comparison (failed before its methods and dataset were resolved)"
"""Stands in for a comparison that failed before it had a scientific description."""


def comparison_label(run: NamedRun | UndescribedComparison) -> str:
    """The text of a comparison's run, or a statement that it was never described."""
    match run:
        case NamedRun():
            return run_name_text(run)
        case UndescribedComparison():
            return UNDESCRIBED_COMPARISON_TEXT


def assignment_label(result: AssignmentResult) -> str:
    """What an assignment trained, as its training run is named."""
    return render_training_label(result.presentation)


def generation_label(result: GenerationResult) -> str:
    """What a generated dataset is."""
    return render_dataset_label(result.dataset, result.notation)
