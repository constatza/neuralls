"""Acceptance checks for the public comparison-metric vocabulary."""

from pathlib import Path

ROOT = Path(__file__).parents[2]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "src/neuralls/domain/solver/README.md",
    ROOT / "src/neuralls/composition/README.md",
    ROOT / "src/neuralls/domain/solver/reporting/__init__.py",
    ROOT / "src/neuralls/platform/reporting/outcome_plots.py",
)


def test_canonical_comparison_terms_are_documented() -> None:
    text = "\n".join(path.read_text() for path in DOCUMENTS)
    for term in (
        "generation",
        "checkpoint",
        "setup",
        "solve",
        "workload online",
        "amortized online",
        "success rate",
        "ECDF",
        "Dolan",
        "paired",
        "speed ratio",
        "matrix cluster",
        "reference",
    ):
        assert term.lower() in text.lower(), term


def test_removed_cost_vocabulary_is_absent_from_user_facing_docs() -> None:
    text = "\n".join(path.read_text() for path in DOCUMENTS)
    for term in ("recurring cost", "shared one-time setup", "recurring_s"):
        assert term not in text
