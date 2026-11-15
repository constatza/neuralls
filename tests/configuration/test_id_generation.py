"""Tests for auto-id and display-name generation helpers."""

from __future__ import annotations

import pytest

from neuralls.platform.config.models.id_generation import (
    _infer_assignment_id,
    _infer_comparison_id,
    _slugify,
    _validate_id_chars,
)

# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "slug"),
    [
        pytest.param("My Experiment Label", "my-experiment-label", id="spaces-to-hyphens"),
        pytest.param("FFNN Standard", "ffnn-standard", id="lowercases"),
        pytest.param("ffnn_standard-v2", "ffnn_standard-v2", id="keeps-hyphens-underscores"),
        pytest.param("  my label  ", "my-label", id="strips-whitespace"),
    ],
)
def test_slugify_gives_the_exact_slug(label: str, slug: str) -> None:
    assert _slugify(label) == slug


@pytest.mark.parametrize(
    "label",
    [
        pytest.param("My 😀 Experiment", id="emoji"),
        pytest.param("Experiment: Run 1", id="colon"),
        pytest.param("", id="empty"),
    ],
)
def test_slugify_rejects_an_unsluggable_label(label: str) -> None:
    with pytest.raises(ValueError):
        _slugify(label)


# ---------------------------------------------------------------------------
# _validate_id_chars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("identifier", ["gaussian-cg0-45x15", "ffnn_standard_v2"])
def test_validate_id_chars_accepts_hyphens_digits_and_underscores(identifier: str) -> None:
    _validate_id_chars(identifier)  # no exception


@pytest.mark.parametrize(
    "identifier",
    [
        pytest.param("has space", id="space"),
        pytest.param("id\U0001f600", id="emoji"),
        pytest.param("id:colon", id="colon"),
    ],
)
def test_validate_id_chars_rejects_other_characters(identifier: str) -> None:
    with pytest.raises(ValueError):
        _validate_id_chars(identifier)


# ---------------------------------------------------------------------------
# _infer_assignment_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_id", "user_dn", "expected"),
    [
        pytest.param(None, None, "ffnn-standard-gaussian-cg0-45x15", id="job-first"),
        pytest.param(None, "My Experiment", "my-experiment", id="from-display-name"),
        pytest.param("my-experiment", None, "my-experiment", id="explicit-id"),
        pytest.param("explicit-id", "Some Display Name", "explicit-id", id="id-beats-display-name"),
    ],
)
def test_infer_assignment_id(user_id: str | None, user_dn: str | None, expected: str) -> None:
    result = _infer_assignment_id(
        dataset_id="gaussian-cg0-45x15",
        job_id="ffnn-standard",
        user_id=user_id,
        user_dn=user_dn,
    )
    assert result == expected


def test_infer_assignment_id_raises_on_invalid_chars() -> None:
    with pytest.raises(ValueError):
        _infer_assignment_id(
            dataset_id="gaussian-cg0-45x15",
            job_id="ffnn-standard",
            user_id="bad id!",
            user_dn=None,
        )


# ---------------------------------------------------------------------------
# _infer_comparison_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("matrix_id", "rhs_id", "user_dn", "expected"),
    [
        pytest.param(
            "gaussian-rhs-45x15", "gaussian-rhs-45x15", None, "gaussian-rhs-45x15", id="same-ids"
        ),
        pytest.param(
            "gaussian-cg0-45x15",
            "solutions-45x15",
            None,
            "gaussian-cg0-45x15-solutions-45x15",
            id="different-ids-joined",
        ),
        pytest.param(
            "gaussian-rhs-45x15",
            "gaussian-rhs-45x15",
            "Gaussian Test",
            "gaussian-test",
            id="from-display-name",
        ),
    ],
)
def test_infer_comparison_id(
    matrix_id: str, rhs_id: str, user_dn: str | None, expected: str
) -> None:
    result = _infer_comparison_id(matrix_id=matrix_id, rhs_id=rhs_id, user_id=None, user_dn=user_dn)
    assert result == expected


def test_infer_comparison_id_raises_on_invalid_chars() -> None:
    with pytest.raises(ValueError):
        _infer_comparison_id(matrix_id="x", rhs_id="x", user_id="bad id!", user_dn=None)
