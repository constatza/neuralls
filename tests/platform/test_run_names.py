"""Run names are target-specific renders of typed presentations, never ids."""

from __future__ import annotations

import pytest

from neuralls.domain.presentation.methods import (
    DatasetQualifiedMethodPresentation,
    StandaloneMethodPresentation,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import (
    ComparisonRunName,
    EditorialComparison,
    EditorialTrainingSubject,
    EvaluationRunName,
    HeldOutMatrixPart,
    MethodRunName,
    RhsDrawPart,
    RunTimestamp,
    SessionRunName,
    SystemQualifier,
    TrainingRunName,
    Untimed,
)
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    EditorialPresentation,
    SessionCounts,
    session_title,
    summarized_session_title,
)
from neuralls.platform.tracking.run_names import render_run_name, render_session_title
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import MatrixId

PLAIN = NotationPolicy.default()
STAMP = RunTimestamp("Mon 26 Oct 2026 - 10:00:00")
SOLE = SystemQualifier(())


def test_session_name_is_the_editorial_title_and_stamp() -> None:
    name = SessionRunName(EditorialPresentation("Spectral comparison"), STAMP)
    assert render_run_name(name, PLAIN) == "Spectral comparison | Mon 26 Oct 2026 - 10:00:00"


def test_session_name_without_editorial_title_is_the_dataset(
    dataset_presentation: DatasetPresentation,
) -> None:
    name = SessionRunName(ComparisonScopePresentation(dataset_presentation), STAMP)
    assert render_run_name(name, PLAIN) == (
        "Test problem | Gaussian CG Residuals (k=20, N=100) | Mon 26 Oct 2026 - 10:00:00"
    )


def test_untimed_comparison_name_is_the_bare_title(
    dataset_presentation: DatasetPresentation,
) -> None:
    title = ComparisonScopePresentation(dataset_presentation)
    text = render_run_name(ComparisonRunName(title, SOLE, Untimed()), PLAIN)
    assert text == "Test problem | Gaussian CG Residuals (k=20, N=100)"


def test_editorial_comparison_name_is_followed_by_its_dataset(
    dataset_presentation: DatasetPresentation,
) -> None:
    subject = EditorialComparison(EditorialPresentation("Spectral study"), dataset_presentation)
    text = render_run_name(ComparisonRunName(subject, SOLE, Untimed()), PLAIN)
    assert text == "Spectral study | Test problem | Gaussian CG Residuals (k=20, N=100)"


def test_sibling_systems_get_distinct_names(dataset_presentation: DatasetPresentation) -> None:
    title = ComparisonScopePresentation(dataset_presentation)
    texts = [
        render_run_name(
            ComparisonRunName(
                title,
                SystemQualifier((HeldOutMatrixPart(MatrixId(3)), RhsDrawPart(rhs))),
                Untimed(),
            ),
            PLAIN,
        )
        for rhs in (0, 1)
    ]
    assert [text.rsplit(" | ", 1)[1] for text in texts] == ["matrix 3, RHS 0", "matrix 3, RHS 1"]


def test_method_name_is_the_method_presentation_with_no_prefix() -> None:
    method = StandaloneMethodPresentation(ScientificDescription("Jacobi"))
    text = render_run_name(MethodRunName(method), PLAIN)
    assert text == "Jacobi"


def test_dataset_trained_method_names_its_dataset_exactly_once(
    dataset_presentation: DatasetPresentation,
) -> None:
    method = DatasetQualifiedMethodPresentation(
        ScientificDescription("POD-2G"), dataset_presentation
    )
    text = render_run_name(MethodRunName(method), PLAIN)
    assert text == "POD-2G | Gaussian CG Residuals (k=20, N=100)"


def test_training_name_of_a_trained_method_carries_its_dataset(
    dataset_presentation: DatasetPresentation,
) -> None:
    subject = DatasetQualifiedMethodPresentation(
        ScientificDescription("POD-2G"), dataset_presentation
    )
    text = render_run_name(TrainingRunName(subject, Untimed()), PLAIN)
    assert text == "POD-2G | Gaussian CG Residuals (k=20, N=100)"


def test_editorial_training_name_appends_the_dataset(
    dataset_presentation: DatasetPresentation,
) -> None:
    subject = EditorialTrainingSubject(EditorialPresentation("Graph net run"), dataset_presentation)
    text = render_run_name(TrainingRunName(subject, STAMP), PLAIN)
    assert text == (
        "Graph net run | Test problem | Gaussian CG Residuals (k=20, N=100)"
        " | Mon 26 Oct 2026 - 10:00:00"
    )


def test_evaluation_name_renders_the_same_subject(
    dataset_presentation: DatasetPresentation,
) -> None:
    subject = EditorialTrainingSubject(EditorialPresentation("Graph net run"), dataset_presentation)
    assert render_run_name(EvaluationRunName(subject), PLAIN) == render_run_name(
        TrainingRunName(subject, Untimed()), PLAIN
    )


@pytest.mark.parametrize("text", ["", "  ", "two\nlines"])
def test_a_timestamp_must_be_one_printable_line(text: str) -> None:
    with pytest.raises(ValueError):
        RunTimestamp(text)


def test_a_session_over_different_titles_lists_each_once_in_a_stable_order(
    dataset_presentation: DatasetPresentation,
) -> None:
    editorial = EditorialPresentation("Spectral comparison")
    scope = ComparisonScopePresentation(dataset_presentation)
    forward = session_title([editorial, scope, editorial])
    backward = session_title([scope, editorial])
    assert render_run_name(SessionRunName(forward, STAMP), PLAIN) == render_run_name(
        SessionRunName(backward, STAMP), PLAIN
    )
    text = render_run_name(SessionRunName(forward, STAMP), PLAIN)
    assert text == (
        f"Spectral comparison; Test problem | Gaussian CG Residuals (k=20, N=100) | {STAMP.text}"
    )


def test_a_many_system_session_is_summarized_by_counts_not_enumerated() -> None:
    families = [EditorialPresentation(name) for name in ("Gaussian RHS", "Sparse Gaussian RHS")]
    titles = [family for family in families for _ in range(10)]
    title = summarized_session_title(titles, SessionCounts(20, 4, 5))
    assert render_run_name(SessionRunName(title, STAMP), PLAIN) == (
        f"20 systems (4 RHS types x 5 matrices) | {STAMP.text}"
    )


def test_a_many_system_session_with_one_shared_title_keeps_it_as_head() -> None:
    shared = EditorialPresentation("Spectral comparison")
    title = summarized_session_title([shared] * 6, SessionCounts(6, 2, 3))
    assert render_run_name(SessionRunName(title, STAMP), PLAIN) == (
        f"Spectral comparison | 6 systems (2 RHS types x 3 matrices) | {STAMP.text}"
    )


def test_the_session_label_matches_the_summarized_run_name_text() -> None:
    shared = EditorialPresentation("Spectral comparison")
    title = summarized_session_title([shared] * 6, SessionCounts(6, 2, 3))
    assert render_session_title(title, PLAIN) == (
        "Spectral comparison | 6 systems (2 RHS types x 3 matrices)"
    )


def test_one_lone_system_keeps_its_own_title() -> None:
    only = EditorialPresentation("Spectral comparison")
    assert summarized_session_title([only], SessionCounts(1, 1, 1)) == only


def test_a_summary_needs_one_title_per_system() -> None:
    with pytest.raises(ValueError, match="one title per system"):
        summarized_session_title([EditorialPresentation("A")], SessionCounts(2, 1, 2))


@pytest.mark.parametrize("counts", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_session_counts_must_be_positive(counts: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="at least one"):
        SessionCounts(*counts)


def test_a_session_over_one_title_has_that_title(dataset_presentation: DatasetPresentation) -> None:
    scope = ComparisonScopePresentation(dataset_presentation)
    assert session_title([scope, scope]) == scope


def test_a_session_without_titles_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one title"):
        session_title([])
