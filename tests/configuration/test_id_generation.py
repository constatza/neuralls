"""Tests for auto-id and display-name generation helpers."""

from __future__ import annotations

import pytest

from neuralls.platform.config.models.id_generation import (
    _build_display_lookup,
    _infer_assignment_display_name,
    _infer_assignment_id,
    _infer_comparison_display_name,
    _infer_comparison_id,
    _slugify,
    _validate_id_chars,
)


@pytest.fixture
def dataset_display() -> dict[str, str]:
    """Display-name lookup for test datasets."""
    return {
        "gaussian-cg0-45x15": "Gaussian CG-1 45x15",
        "solutions-45x15": "Solutions 45x15",
    }


@pytest.fixture
def model_display() -> dict[str, str]:
    """Display-name lookup for test jobs."""
    return {
        "ffnn-standard": "FFNN Standard",
        "ffnn-large": "FFNN Large",
    }


@pytest.fixture
def exp_invalid_id() -> dict[str, object]:
    """Assignment entry with an id that contains invalid characters."""
    return {"dataset": "gaussian-cg0-45x15", "job": "ffnn-standard", "id": "bad id!"}


@pytest.fixture
def exp_unknown_registry() -> dict[str, object]:
    """Assignment entry whose dataset/job are absent from the display lookups."""
    return {"dataset": "unknown-ds", "job": "unknown-job"}


@pytest.fixture
def comp_same_datasets() -> dict[str, object]:
    """Comparison with identical matrix and RHS source ids, no id or display_name."""
    return {"matrix_dataset": "gaussian-rhs-45x15", "rhs_source": "gaussian-rhs-45x15"}


@pytest.fixture
def comp_diff_datasets() -> dict[str, object]:
    """Comparison with different matrix and RHS source ids, no id or display_name."""
    return {"matrix_dataset": "gaussian-cg0-45x15", "rhs_source": "solutions-45x15"}


@pytest.fixture
def comp_same_display_name_test_data() -> dict[str, object]:
    """Comparison with same matrix/RHS source id used to test auto display name."""
    return {"matrix_dataset": "gaussian-cg0-45x15", "rhs_source": "gaussian-cg0-45x15"}


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


def test_slugify_converts_spaces_to_hyphens() -> None:
    assert _slugify("My Experiment Label") == "my-experiment-label"


def test_slugify_lowercases_input() -> None:
    assert _slugify("FFNN Standard") == "ffnn-standard"


def test_slugify_preserves_existing_hyphens_and_underscores() -> None:
    assert _slugify("ffnn_standard-v2") == "ffnn_standard-v2"


def test_slugify_strips_surrounding_whitespace() -> None:
    assert _slugify("  my label  ") == "my-label"


def test_slugify_raises_on_emoji() -> None:
    with pytest.raises(ValueError, match="invalid"):
        _slugify("My 😀 Experiment")


def test_slugify_raises_on_colon() -> None:
    with pytest.raises(ValueError, match="invalid"):
        _slugify("Experiment: Run 1")


def test_slugify_raises_on_empty_string() -> None:
    with pytest.raises(ValueError):
        _slugify("")


# ---------------------------------------------------------------------------
# _validate_id_chars
# ---------------------------------------------------------------------------


def test_validate_id_chars_accepts_hyphens_and_digits() -> None:
    _validate_id_chars("gaussian-cg0-45x15")  # no exception


def test_validate_id_chars_accepts_underscores() -> None:
    _validate_id_chars("ffnn_standard_v2")  # no exception


def test_validate_id_chars_rejects_space() -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_id_chars("has space")


def test_validate_id_chars_rejects_emoji() -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_id_chars("id\U0001f600")


def test_validate_id_chars_rejects_colon() -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_id_chars("id:colon")


# ---------------------------------------------------------------------------
# _build_display_lookup
# ---------------------------------------------------------------------------


def test_build_display_lookup_returns_display_name_when_present() -> None:
    entries: list[object] = [{"id": "ds1", "display_name": "Dataset One"}]
    assert _build_display_lookup(entries) == {"ds1": "Dataset One"}


def test_build_display_lookup_falls_back_to_id_when_no_display_name() -> None:
    entries: list[object] = [{"id": "ds1"}]
    assert _build_display_lookup(entries) == {"ds1": "ds1"}


def test_build_display_lookup_falls_back_to_id_when_display_name_blank() -> None:
    entries: list[object] = [{"id": "ds1", "display_name": "  "}]
    assert _build_display_lookup(entries) == {"ds1": "ds1"}


def test_build_display_lookup_ignores_non_dict_entries() -> None:
    entries: list[object] = ["not-a-dict", None, 42]
    assert _build_display_lookup(entries) == {}


def test_build_display_lookup_handles_mixed_entries() -> None:
    entries: list[object] = [
        {"id": "ds1", "display_name": "Dataset One"},
        {"id": "ds2"},
        {"id": "ds3", "display_name": "  "},
    ]
    assert _build_display_lookup(entries) == {
        "ds1": "Dataset One",
        "ds2": "ds2",
        "ds3": "ds3",
    }


# ---------------------------------------------------------------------------
# _infer_assignment_id
# ---------------------------------------------------------------------------


def test_infer_assignment_id_auto_is_job_first() -> None:
    result = _infer_assignment_id(
        dataset_id="gaussian-cg0-45x15",
        job_id="ffnn-standard",
        user_id=None,
        user_dn=None,
    )
    assert result == "ffnn-standard-gaussian-cg0-45x15"


def test_infer_assignment_id_derives_from_display_name() -> None:
    result = _infer_assignment_id(
        dataset_id="gaussian-cg0-45x15",
        job_id="ffnn-standard",
        user_id=None,
        user_dn="My Experiment",
    )
    assert result == "my-experiment"


def test_infer_assignment_id_uses_explicit_user_id() -> None:
    result = _infer_assignment_id(
        dataset_id="gaussian-cg0-45x15",
        job_id="ffnn-standard",
        user_id="my-experiment",
        user_dn=None,
    )
    assert result == "my-experiment"


def test_infer_assignment_id_user_id_takes_priority_over_display_name() -> None:
    result = _infer_assignment_id(
        dataset_id="gaussian-cg0-45x15",
        job_id="ffnn-standard",
        user_id="explicit-id",
        user_dn="Some Display Name",
    )
    assert result == "explicit-id"


def test_infer_assignment_id_raises_on_invalid_chars(
    exp_invalid_id: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        _infer_assignment_id(
            dataset_id=str(exp_invalid_id["dataset"]),
            job_id=str(exp_invalid_id["job"]),
            user_id=str(exp_invalid_id["id"]),
            user_dn=None,
        )


def test_infer_assignment_id_falls_back_to_raw_id_when_not_in_registry(
    exp_unknown_registry: dict[str, object],
) -> None:
    result = _infer_assignment_id(
        dataset_id=str(exp_unknown_registry["dataset"]),
        job_id=str(exp_unknown_registry["job"]),
        user_id=None,
        user_dn=None,
    )
    assert result == "unknown-job-unknown-ds"


# ---------------------------------------------------------------------------
# _infer_assignment_display_name
# ---------------------------------------------------------------------------


def test_infer_assignment_display_name_auto_from_registry(
    dataset_display: dict[str, str],
    model_display: dict[str, str],
) -> None:
    result = _infer_assignment_display_name(
        dataset_id="gaussian-cg0-45x15",
        job_id="ffnn-standard",
        user_dn=None,
        dataset_display=dataset_display,
        model_display=model_display,
    )
    assert result == "FFNN Standard | Gaussian CG-1 45x15"


def test_infer_assignment_display_name_uses_explicit_user_dn(
    dataset_display: dict[str, str],
    model_display: dict[str, str],
) -> None:
    result = _infer_assignment_display_name(
        dataset_id="gaussian-cg0-45x15",
        job_id="ffnn-standard",
        user_dn="My Experiment",
        dataset_display=dataset_display,
        model_display=model_display,
    )
    assert result == "My Experiment"


def test_infer_assignment_display_name_falls_back_to_raw_id_when_not_in_registry(
    exp_unknown_registry: dict[str, object],
    dataset_display: dict[str, str],
    model_display: dict[str, str],
) -> None:
    result = _infer_assignment_display_name(
        dataset_id=str(exp_unknown_registry["dataset"]),
        job_id=str(exp_unknown_registry["job"]),
        user_dn=None,
        dataset_display=dataset_display,
        model_display=model_display,
    )
    assert result == "unknown-job | unknown-ds"


def test_infer_assignment_display_name_auto_when_only_id_given(
    dataset_display: dict[str, str],
    model_display: dict[str, str],
) -> None:
    result = _infer_assignment_display_name(
        dataset_id="gaussian-cg0-45x15",
        job_id="ffnn-standard",
        user_dn=None,
        dataset_display=dataset_display,
        model_display=model_display,
    )
    assert result == "FFNN Standard | Gaussian CG-1 45x15"


# ---------------------------------------------------------------------------
# _infer_comparison_id
# ---------------------------------------------------------------------------


def test_infer_comparison_id_uses_matrix_id_when_datasets_same(
    comp_same_datasets: dict[str, object],
) -> None:
    result = _infer_comparison_id(
        matrix_id=str(comp_same_datasets["matrix_dataset"]),
        rhs_id=str(comp_same_datasets["rhs_source"]),
        user_id=None,
        user_dn=None,
    )
    assert result == "gaussian-rhs-45x15"


def test_infer_comparison_id_joins_ids_when_datasets_differ(
    comp_diff_datasets: dict[str, object],
) -> None:
    result = _infer_comparison_id(
        matrix_id=str(comp_diff_datasets["matrix_dataset"]),
        rhs_id=str(comp_diff_datasets["rhs_source"]),
        user_id=None,
        user_dn=None,
    )
    assert result == "gaussian-cg0-45x15-solutions-45x15"


def test_infer_comparison_id_derives_from_display_name(
    comp_same_datasets: dict[str, object],
) -> None:
    result = _infer_comparison_id(
        matrix_id=str(comp_same_datasets["matrix_dataset"]),
        rhs_id=str(comp_same_datasets["rhs_source"]),
        user_id=None,
        user_dn="Gaussian Test",
    )
    assert result == "gaussian-test"


def test_infer_comparison_id_raises_on_invalid_chars(
    dataset_display: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        _infer_comparison_id(
            matrix_id="x",
            rhs_id="x",
            user_id="bad id!",
            user_dn=None,
        )


# ---------------------------------------------------------------------------
# _infer_comparison_display_name
# ---------------------------------------------------------------------------


def test_infer_comparison_display_name_when_same_datasets(
    comp_same_display_name_test_data: dict[str, object],
    dataset_display: dict[str, str],
) -> None:
    result = _infer_comparison_display_name(
        matrix_id=str(comp_same_display_name_test_data["matrix_dataset"]),
        rhs_id=str(comp_same_display_name_test_data["rhs_source"]),
        user_dn=None,
        dataset_display=dataset_display,
    )
    assert result == "Gaussian CG-1 45x15"


def test_infer_comparison_display_name_when_datasets_differ(
    comp_diff_datasets: dict[str, object],
    dataset_display: dict[str, str],
) -> None:
    result = _infer_comparison_display_name(
        matrix_id=str(comp_diff_datasets["matrix_dataset"]),
        rhs_id=str(comp_diff_datasets["rhs_source"]),
        user_dn=None,
        dataset_display=dataset_display,
    )
    assert result == "Gaussian CG-1 45x15 | Solutions 45x15"


def test_infer_comparison_display_name_uses_explicit_user_dn(
    comp_same_datasets: dict[str, object],
    dataset_display: dict[str, str],
) -> None:
    result = _infer_comparison_display_name(
        matrix_id=str(comp_same_datasets["matrix_dataset"]),
        rhs_id=str(comp_same_datasets["rhs_source"]),
        user_dn="Gaussian Test",
        dataset_display=dataset_display,
    )
    assert result == "Gaussian Test"


def test_infer_comparison_display_name_is_dataset_only(
    comp_diff_datasets: dict[str, object],
    dataset_display: dict[str, str],
) -> None:
    result = _infer_comparison_display_name(
        matrix_id=str(comp_diff_datasets["matrix_dataset"]),
        rhs_id=str(comp_diff_datasets["rhs_source"]),
        user_dn=None,
        dataset_display=dataset_display,
    )
    assert result == "Gaussian CG-1 45x15 | Solutions 45x15"
