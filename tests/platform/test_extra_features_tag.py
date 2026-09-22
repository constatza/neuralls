"""Regression tests for log_extra_feature_names_tag declaration-order preservation.

Bug fixed: tag_value was built with ``sorted(extra_names)`` instead of
``",".join(extra_names)``, destroying positional ordering of feature names.
Since parameter files are stored as ``parameters_0.npy``, ``parameters_1.npy``
in declaration order, losing the order caused wrong arrays to be bound at
comparison time.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from neuralls.platform.tracking.extra_features import (
    EXTRA_FEATURE_NAMES_TAG,
    fetch_extra_feature_names,
    log_extra_feature_names_tag,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRACKING_URI: str = "sqlite:///test.db"
_RUN_ID: str = "run-abc-123"

# Non-alphabetical names: alphabetical sort would give apple < zebra,
# reversing the intended declaration order.
_NON_ALPHA_NAMES: tuple[str, ...] = ("zebra", "apple")
_ALPHA_NAMES: tuple[str, ...] = ("apple", "zebra")
_SINGLE_NAME: tuple[str, ...] = ("only_one",)
_EMPTY_NAMES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mlflow_client() -> MagicMock:
    """MagicMock for MlflowClient with autospec enforcement.

    Returns:
        Autospec-based mock of MlflowClient.
    """
    from mlflow.tracking import MlflowClient

    return MagicMock(spec=MlflowClient)


def _make_run_stub(tag_value: str) -> SimpleNamespace:
    """Build a minimal MLflow run stub carrying a single tag.

    Args:
        tag_value: Raw string value stored under EXTRA_FEATURE_NAMES_TAG.

    Returns:
        SimpleNamespace mimicking ``MlflowClient.get_run()`` return value.
    """
    return SimpleNamespace(
        data=SimpleNamespace(
            tags={EXTRA_FEATURE_NAMES_TAG: tag_value},
        )
    )


@pytest.fixture
def run_stub_non_alpha() -> SimpleNamespace:
    """Run stub with non-alphabetical tag value ``"zebra,apple"``."""
    return _make_run_stub(",".join(_NON_ALPHA_NAMES))


@pytest.fixture
def run_stub_alpha() -> SimpleNamespace:
    """Run stub with alphabetical tag value ``"apple,zebra"``."""
    return _make_run_stub(",".join(_ALPHA_NAMES))


@pytest.fixture
def run_stub_single() -> SimpleNamespace:
    """Run stub with a single feature name."""
    return _make_run_stub(",".join(_SINGLE_NAME))


@pytest.fixture
def run_stub_empty_tag() -> SimpleNamespace:
    """Run stub with an absent tag (empty string default)."""
    return _make_run_stub("")


@pytest.fixture
def experiment_stub() -> SimpleNamespace:
    """Single MLflow experiment stub carrying only an id."""
    return SimpleNamespace(experiment_id="0")


def _make_run_id_stub(run_id: str) -> SimpleNamespace:
    """Build a minimal MLflow run stub carrying only a run id.

    Args:
        run_id: Run UUID to expose via ``.info.run_id``.

    Returns:
        SimpleNamespace mimicking a ``search_runs()`` result entry.
    """
    return SimpleNamespace(info=SimpleNamespace(run_id=run_id))


# ---------------------------------------------------------------------------
# Tests for log_extra_feature_names_tag
# ---------------------------------------------------------------------------


def test_log_tag_joins_names_in_declaration_order(mock_mlflow_client: MagicMock) -> None:
    """log_extra_feature_names_tag must write names in the order provided.

    Regression: the old implementation called ``sorted(extra_names)``
    before joining, which silently reordered ``("zebra", "apple")`` to
    ``("apple", "zebra")``.
    """
    with patch(
        "neuralls.platform.tracking.extra_features.MlflowClient",
        return_value=mock_mlflow_client,
    ):
        log_extra_feature_names_tag(_TRACKING_URI, _RUN_ID, _NON_ALPHA_NAMES)

    mock_mlflow_client.set_tag.assert_called_once_with(
        _RUN_ID,
        EXTRA_FEATURE_NAMES_TAG,
        "zebra,apple",
    )


def test_log_tag_alphabetical_input_is_unchanged(mock_mlflow_client: MagicMock) -> None:
    """Alphabetical input must survive the round-trip unchanged.

    This is the trivial path that passed even with the buggy sorted() call.
    It is included for completeness and to document the contract.
    """
    with patch(
        "neuralls.platform.tracking.extra_features.MlflowClient",
        return_value=mock_mlflow_client,
    ):
        log_extra_feature_names_tag(_TRACKING_URI, _RUN_ID, _ALPHA_NAMES)

    mock_mlflow_client.set_tag.assert_called_once_with(
        _RUN_ID,
        EXTRA_FEATURE_NAMES_TAG,
        "apple,zebra",
    )


def test_log_tag_single_name_no_comma(mock_mlflow_client: MagicMock) -> None:
    """A single feature name must be stored without a comma."""
    with patch(
        "neuralls.platform.tracking.extra_features.MlflowClient",
        return_value=mock_mlflow_client,
    ):
        log_extra_feature_names_tag(_TRACKING_URI, _RUN_ID, _SINGLE_NAME)

    mock_mlflow_client.set_tag.assert_called_once_with(
        _RUN_ID,
        EXTRA_FEATURE_NAMES_TAG,
        "only_one",
    )


def test_log_tag_empty_names_stores_empty_string(mock_mlflow_client: MagicMock) -> None:
    """Empty iterable must result in an empty tag value (no crash)."""
    with patch(
        "neuralls.platform.tracking.extra_features.MlflowClient",
        return_value=mock_mlflow_client,
    ):
        log_extra_feature_names_tag(_TRACKING_URI, _RUN_ID, _EMPTY_NAMES)

    mock_mlflow_client.set_tag.assert_called_once_with(
        _RUN_ID,
        EXTRA_FEATURE_NAMES_TAG,
        "",
    )


# ---------------------------------------------------------------------------
# Tests for fetch_extra_feature_names
# ---------------------------------------------------------------------------


def test_fetch_returns_names_in_stored_order(
    mock_mlflow_client: MagicMock,
    run_stub_non_alpha: SimpleNamespace,
) -> None:
    """fetch_extra_feature_names must return the tuple in stored tag order.

    The tag stores ``"zebra,apple"``; the result must be ``("zebra", "apple")``,
    not sorted to ``("apple", "zebra")``.
    """
    mock_mlflow_client.get_run.return_value = run_stub_non_alpha
    result = fetch_extra_feature_names(_RUN_ID, client=mock_mlflow_client)
    assert result == ("zebra", "apple")


def test_fetch_returns_alphabetical_order_when_tag_is_alphabetical(
    mock_mlflow_client: MagicMock,
    run_stub_alpha: SimpleNamespace,
) -> None:
    """fetch_extra_feature_names returns alphabetical tuple when stored as such."""
    mock_mlflow_client.get_run.return_value = run_stub_alpha
    result = fetch_extra_feature_names(_RUN_ID, client=mock_mlflow_client)
    assert result == ("apple", "zebra")


def test_fetch_returns_single_name_tuple(
    mock_mlflow_client: MagicMock,
    run_stub_single: SimpleNamespace,
) -> None:
    """Single-name tag must yield a one-element tuple."""
    mock_mlflow_client.get_run.return_value = run_stub_single
    result = fetch_extra_feature_names(_RUN_ID, client=mock_mlflow_client)
    assert result == ("only_one",)


def test_fetch_returns_empty_tuple_when_tag_absent(
    mock_mlflow_client: MagicMock,
    run_stub_empty_tag: SimpleNamespace,
) -> None:
    """Absent or empty tag must yield an empty tuple."""
    mock_mlflow_client.get_run.return_value = run_stub_empty_tag
    result = fetch_extra_feature_names(_RUN_ID, client=mock_mlflow_client)
    assert result == ()


# ---------------------------------------------------------------------------
# Round-trip: log then fetch must recover declaration order
# ---------------------------------------------------------------------------


def test_round_trip_preserves_non_alphabetical_declaration_order() -> None:
    """Round-trip log → fetch must preserve non-alphabetical declaration order.

    This is the core regression test. The old ``sorted()`` call would have
    turned ``("zebra", "apple")`` into ``"apple,zebra"`` in the tag, causing
    ``fetch_extra_feature_names`` to return the wrong order.
    """
    # Capture the tag value written by log_extra_feature_names_tag.
    captured_tag: dict[str, str] = {}

    def _fake_set_tag(run_id: str, key: str, value: str) -> None:
        captured_tag[key] = value

    log_client = MagicMock()
    log_client.set_tag.side_effect = _fake_set_tag

    with patch(
        "neuralls.platform.tracking.extra_features.MlflowClient",
        return_value=log_client,
    ):
        log_extra_feature_names_tag(_TRACKING_URI, _RUN_ID, _NON_ALPHA_NAMES)

    # Build a fetch client that returns the tag value written above.
    fetch_client = MagicMock()
    fetch_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(tags={EXTRA_FEATURE_NAMES_TAG: captured_tag[EXTRA_FEATURE_NAMES_TAG]})
    )
    result = fetch_extra_feature_names(_RUN_ID, client=fetch_client)

    assert result == _NON_ALPHA_NAMES, (
        f"Expected {_NON_ALPHA_NAMES!r} but got {result!r}. "
        "The old sorted() call would have produced ('apple', 'zebra')."
    )


def test_round_trip_sorted_names_would_be_different_from_declaration_order() -> None:
    """Guard: sorted result is provably different from declaration order.

    This test documents why the bug mattered: for non-alphabetical input,
    sorted() produces a different sequence. If this assertion fails, the
    names chosen for the test are accidentally already alphabetical, making
    the regression test vacuous.
    """
    assert tuple(sorted(_NON_ALPHA_NAMES)) != _NON_ALPHA_NAMES, (
        "Test data must be non-alphabetical for the regression to be meaningful."
    )
