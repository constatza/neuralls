"""Integration tests for CaseConfig auto-id and display-name filling.

These tests verify that the CaseConfig validator correctly infers missing id and
display_name fields in AssignmentEntry and ComparisonRegistryEntry, using the
registry entries (datasets and jobs) to fill in labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.experiments import CaseConfig


@pytest.fixture
def dataset_entry_gaussian() -> dict[str, object]:
    """Registry entry for Gaussian CG-1 45x15 dataset."""
    return {
        "id": "gaussian-cg0-45x15",
        "path": Path("fake/gaussian.toml"),
        "display_name": "Gaussian CG-1 45x15",
    }


@pytest.fixture
def dataset_entry_solutions() -> dict[str, object]:
    """Registry entry for Solutions 45x15 dataset."""
    return {
        "id": "solutions-45x15",
        "path": Path("fake/solutions.toml"),
        "display_name": "Solutions 45x15",
    }


@pytest.fixture
def job_entry_ffnn() -> dict[str, object]:
    """Registry entry for FFNN Standard job."""
    return {
        "id": "ffnn-standard",
        "path": Path("fake/ffnn.toml"),
        "display_name": "FFNN Standard",
    }


@pytest.fixture
def minimal_case_raw(
    dataset_entry_gaussian: dict[str, object],
    job_entry_ffnn: dict[str, object],
) -> dict[str, object]:
    """Minimal valid case config with registry entries but empty assignments/comparisons."""
    return {
        "datasets": [dataset_entry_gaussian],
        "jobs": [job_entry_ffnn],
        "assignments": [],
        "comparisons": [],
    }


# ============================================================================
# Assignments — auto-fill tests
# ============================================================================


@pytest.mark.parametrize(
    ("entry", "expected_id", "expected_display_name"),
    [
        pytest.param(
            {"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard"},
            "ffnn-standard-gaussian-cg0-45x15",
            None,
            id="auto-id-is-job-first",
        ),
        pytest.param(
            {"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard", "display_name": "My Run"},
            "my-run",
            "My Run",
            id="id-slugified-from-display-name",
        ),
        pytest.param(
            {"id": "my-id", "dataset": "gaussian-cg0-45x15", "job": "ffnn-standard"},
            "my-id",
            None,
            id="display-name-never-rebuilt-from-ids",
        ),
    ],
)
def test_assignment_auto_fill(
    minimal_case_raw: dict[str, object],
    entry: dict[str, object],
    expected_id: str,
    expected_display_name: str | None,
) -> None:
    config = CaseConfig.model_validate({**minimal_case_raw, "assignments": [entry]})

    assert config.assignments[0].id == expected_id
    assert config.assignments[0].display_name == expected_display_name


# ============================================================================
# Comparisons — auto-fill tests
# ============================================================================


@pytest.mark.parametrize(
    ("rhs_kind", "expected_id"),
    [
        pytest.param("gaussian", "gaussian-cg0-45x15-gaussian", id="gaussian-rhs"),
        pytest.param("dataset", "gaussian-cg0-45x15-dataset", id="dataset-rhs"),
    ],
)
def test_comparison_auto_id_joins_matrix_dataset_and_rhs_kind(
    minimal_case_raw: dict[str, object], rhs_kind: str, expected_id: str
) -> None:
    raw = dict(minimal_case_raw)
    raw["comparisons"] = [
        {
            "matrix_dataset": "gaussian-cg0-45x15",
            "rhs_source": {"kind": rhs_kind},
            "assignments": ["ffnn-standard-gaussian-cg0-45x15"],
        }
    ]
    raw["assignments"] = [{"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard"}]
    config = CaseConfig.model_validate(raw)
    assert config.comparisons[0].id == expected_id
    assert config.comparisons[0].display_name is None


# ============================================================================
# Backward compatibility tests
# ============================================================================


def test_explicit_id_and_display_name_preserved(
    minimal_case_raw: dict[str, object],
) -> None:
    """Explicit id and display_name in both assignments and comparisons are preserved."""
    raw = dict(minimal_case_raw)
    raw["assignments"] = [
        {
            "id": "my-exp",
            "dataset": "gaussian-cg0-45x15",
            "job": "ffnn-standard",
            "display_name": "My Assignment",
        }
    ]
    raw["comparisons"] = [
        {
            "id": "my-comp",
            "matrix_dataset": "gaussian-cg0-45x15",
            "rhs_source": {"kind": "gaussian"},
            "display_name": "My Comparison",
        }
    ]
    config = CaseConfig.model_validate(raw)

    assert config.assignments[0].id == "my-exp"
    assert config.assignments[0].display_name == "My Assignment"

    assert config.comparisons[0].id == "my-comp"
    assert config.comparisons[0].display_name == "My Comparison"


# ============================================================================
# Validation error tests
# ============================================================================


def test_invalid_id_chars_raise_validation_error(
    minimal_case_raw: dict[str, object],
) -> None:
    """Reject assignment entries with invalid characters in auto-generated id."""
    raw = dict(minimal_case_raw)
    raw["assignments"] = [
        {
            "id": "bad id!",
            "dataset": "gaussian-cg0-45x15",
            "job": "ffnn-standard",
        }
    ]
    with pytest.raises(ValidationError):
        CaseConfig.model_validate(raw)


def test_multi_assignment_auto_ids_all_job_first(
    minimal_case_raw: dict[str, object],
    dataset_entry_solutions: dict[str, object],
) -> None:
    """All auto-generated ids across multiple datasets are job-first.

    Mirrors the production config pattern: datasets with display_names, one job,
    multiple assignments with only dataset + job (no id or display_name).
    """
    raw = dict(minimal_case_raw)
    raw["datasets"] = [*list(raw["datasets"]), dataset_entry_solutions]  # type: ignore[arg-type]
    raw["assignments"] = [
        {"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard"},
        {"dataset": "solutions-45x15", "job": "ffnn-standard"},
    ]
    config = CaseConfig.model_validate(raw)
    assert [e.id for e in config.assignments] == [
        "ffnn-standard-gaussian-cg0-45x15",
        "ffnn-standard-solutions-45x15",
    ]
    assert [e.display_name for e in config.assignments] == [None, None]


def test_comparison_display_name_ignores_assignment_filter_for_parent_name(
    minimal_case_raw: dict[str, object],
    dataset_entry_solutions: dict[str, object],
) -> None:
    """Comparison parent names stay dataset-defined even with filtered assignments."""
    raw = dict(minimal_case_raw)
    raw["datasets"] = [*list(raw["datasets"]), dataset_entry_solutions]  # type: ignore[arg-type]
    raw["jobs"] = [
        *list(raw["jobs"]),  # type: ignore[arg-type]
        {
            "id": "ffnn-large",
            "path": Path("fake/ffnn-large.toml"),
            "display_name": "FFNN Large",
        },
    ]
    raw["assignments"] = [
        {"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard"},
        {"dataset": "solutions-45x15", "job": "ffnn-large"},
    ]
    raw["comparisons"] = [
        {
            "matrix_dataset": "gaussian-cg0-45x15",
            "rhs_source": {"kind": "dataset"},
            "assignments": ["ffnn-standard-gaussian-cg0-45x15"],
        }
    ]

    config = CaseConfig.model_validate(raw)

    assert config.comparisons[0].display_name is None


def test_duplicate_auto_ids_raise_validation_error(
    minimal_case_raw: dict[str, object],
) -> None:
    """Reject assignment entries when auto-id generation results in duplicates.

    Two assignments with the same dataset and job but no explicit id will
    both auto-generate the same id, which should be rejected.
    """
    raw = dict(minimal_case_raw)
    raw["assignments"] = [
        {
            "dataset": "gaussian-cg0-45x15",
            "job": "ffnn-standard",
            # auto-generates "ffnn-standard-gaussian-cg0-45x15"
        },
        {
            "dataset": "gaussian-cg0-45x15",
            "job": "ffnn-standard",
            # also auto-generates "ffnn-standard-gaussian-cg0-45x15" -> duplicate!
        },
    ]
    with pytest.raises(ValidationError):
        CaseConfig.model_validate(raw)
