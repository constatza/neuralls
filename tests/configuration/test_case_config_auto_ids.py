"""Integration tests for CaseConfig auto-id and display-name filling.

These tests verify that the CaseConfig validator correctly infers missing id and
display_name fields in AssignmentEntry and ComparisonRegistryEntry, using the
registry entries (datasets and jobs) to fill in labels.

Note: These tests are expected to FAIL before the validator is added to CaseConfig.
They should raise pydantic.ValidationError when AssignmentEntry.id or
ComparisonRegistryEntry.id are required but missing.
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
        "path": Path("/fake/gaussian.toml"),
        "display_name": "Gaussian CG-1 45x15",
    }


@pytest.fixture
def dataset_entry_solutions() -> dict[str, object]:
    """Registry entry for Solutions 45x15 dataset."""
    return {
        "id": "solutions-45x15",
        "path": Path("/fake/solutions.toml"),
        "display_name": "Solutions 45x15",
    }


@pytest.fixture
def job_entry_ffnn() -> dict[str, object]:
    """Registry entry for FFNN Standard job."""
    return {
        "id": "ffnn-standard",
        "path": Path("/fake/ffnn.toml"),
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


def test_assignment_auto_id_is_job_first(
    minimal_case_raw: dict[str, object],
) -> None:
    """Auto-generate assignment id from job and dataset when no id or display_name given.

    The id should be "{job_id}-{dataset_id}" (job first).
    Display name should be "{job_label} | {dataset_label}".
    """
    raw = dict(minimal_case_raw)
    raw["assignments"] = [
        {
            "dataset": "gaussian-cg0-45x15",
            "job": "ffnn-standard",
            # no id, no display_name
        }
    ]
    config = CaseConfig.model_validate(raw)
    assert config.assignments[0].id == "ffnn-standard-gaussian-cg0-45x15"
    assert config.assignments[0].display_name == "FFNN Standard | Gaussian CG-1 45x15"


def test_assignment_id_derived_from_display_name(
    minimal_case_raw: dict[str, object],
) -> None:
    """Auto-generate assignment id from display_name via slugification when no id given."""
    raw = dict(minimal_case_raw)
    raw["assignments"] = [
        {
            "dataset": "gaussian-cg0-45x15",
            "job": "ffnn-standard",
            "display_name": "My Run",
            # no id
        }
    ]
    config = CaseConfig.model_validate(raw)
    assert config.assignments[0].id == "my-run"
    assert config.assignments[0].display_name == "My Run"


def test_assignment_display_name_auto_filled_when_only_id_given(
    minimal_case_raw: dict[str, object],
) -> None:
    """Auto-fill display_name from dataset and model labels when only id is given."""
    raw = dict(minimal_case_raw)
    raw["assignments"] = [
        {
            "id": "my-id",
            "dataset": "gaussian-cg0-45x15",
            "job": "ffnn-standard",
            # no display_name
        }
    ]
    config = CaseConfig.model_validate(raw)
    assert config.assignments[0].id == "my-id"
    assert config.assignments[0].display_name == "FFNN Standard | Gaussian CG-1 45x15"


def test_assignment_explicit_id_and_display_name_unchanged(
    minimal_case_raw: dict[str, object],
) -> None:
    """Explicit id and display_name should remain unchanged."""
    raw = dict(minimal_case_raw)
    raw["assignments"] = [
        {
            "id": "my-id",
            "dataset": "gaussian-cg0-45x15",
            "job": "ffnn-standard",
            "display_name": "My Label",
        }
    ]
    config = CaseConfig.model_validate(raw)
    assert config.assignments[0].id == "my-id"
    assert config.assignments[0].display_name == "My Label"


# ============================================================================
# Comparisons — auto-fill tests
# ============================================================================


def test_comparison_auto_id_when_same_datasets(
    minimal_case_raw: dict[str, object],
) -> None:
    """Auto-generate comparison id from matrix dataset and RHS source kind."""
    raw = dict(minimal_case_raw)
    raw["comparisons"] = [
        {
            "matrix_dataset": "gaussian-cg0-45x15",
            "rhs_source": {"kind": "gaussian"},
            "assignments": ["ffnn-standard-gaussian-cg0-45x15"],
            # no id, no display_name
        }
    ]
    raw["assignments"] = [{"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard"}]
    config = CaseConfig.model_validate(raw)
    assert config.comparisons[0].id == "gaussian-cg0-45x15-gaussian"
    assert config.comparisons[0].display_name == "Gaussian CG-1 45x15 | gaussian"
    assert config.comparisons[0].matrix_index == 0


def test_comparison_auto_id_when_different_datasets(
    minimal_case_raw: dict[str, object],
) -> None:
    """Auto-generate comparison id with matrix dataset and dataset RHS source."""
    raw = dict(minimal_case_raw)
    raw["comparisons"] = [
        {
            "matrix_dataset": "gaussian-cg0-45x15",
            "rhs_source": {"kind": "dataset", "path": "/fake/solutions"},
            "assignments": ["ffnn-standard-gaussian-cg0-45x15"],
            # no id, no display_name
        }
    ]
    raw["assignments"] = [{"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard"}]
    config = CaseConfig.model_validate(raw)
    assert config.comparisons[0].id == "gaussian-cg0-45x15-dataset"
    assert config.comparisons[0].display_name == "Gaussian CG-1 45x15 | dataset"


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


def test_comparison_explicit_indices_are_preserved(
    minimal_case_raw: dict[str, object],
) -> None:
    """Comparison entries keep explicit matrix and RHS-source sample indices."""
    from neuralls.platform.config.models.comparison import DatasetRhsSourceModel

    raw = dict(minimal_case_raw)
    raw["assignments"] = [{"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard"}]
    raw["comparisons"] = [
        {
            "matrix_dataset": "gaussian-cg0-45x15",
            "matrix_index": 3,
            "rhs_source": {
                "kind": "dataset",
                "path": "/fake/gaussian-cg0-45x15",
                "sample_index": 7,
            },
        }
    ]
    config = CaseConfig.model_validate(raw)
    assert config.comparisons[0].matrix_index == 3
    assert isinstance(config.comparisons[0].rhs_source, DatasetRhsSourceModel)
    assert config.comparisons[0].rhs_source.sample_index == 7


def test_comparison_rejects_removed_train_run_id(
    minimal_case_raw: dict[str, object],
) -> None:
    """Case comparisons no longer accept split-driven train_run_id metadata."""
    raw = dict(minimal_case_raw)
    raw["assignments"] = [{"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard"}]
    raw["comparisons"] = [
        {
            "matrix_dataset": "gaussian-cg0-45x15",
            "rhs_source": {"kind": "gaussian"},
            "train_run_id": "run-123",
        }
    ]
    with pytest.raises(ValidationError, match="train_run_id"):
        CaseConfig.model_validate(raw)


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
    assert [e.display_name for e in config.assignments] == [
        "FFNN Standard | Gaussian CG-1 45x15",
        "FFNN Standard | Solutions 45x15",
    ]


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
            "path": Path("/fake/ffnn-large.toml"),
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
            "rhs_source": {"kind": "dataset", "path": "/fake/solutions"},
            "assignments": ["ffnn-standard-gaussian-cg0-45x15"],
        }
    ]

    config = CaseConfig.model_validate(raw)

    assert config.comparisons[0].display_name == "Gaussian CG-1 45x15 | dataset"


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
    with pytest.raises(ValidationError, match="[Dd]uplicate"):
        CaseConfig.model_validate(raw)
