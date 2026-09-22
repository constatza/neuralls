"""Tests for mark_run_failed's best-effort status update."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from neuralls.platform.tracking.mlflow_client import mark_run_failed


def test_mark_run_failed_calls_set_terminated_with_failed_status() -> None:
    """mark_run_failed terminates the given run with status FAILED."""
    client = MagicMock()
    with patch("mlflow.tracking.MlflowClient", return_value=client) as mlflow_client_cls:
        mark_run_failed(run_id="run-abc", tracking_uri="sqlite:///tracking.db")
    mlflow_client_cls.assert_called_once_with(tracking_uri="sqlite:///tracking.db")
    client.set_terminated.assert_called_once_with("run-abc", status="FAILED")


def test_mark_run_failed_swallows_exceptions_from_set_terminated() -> None:
    """mark_run_failed is best-effort — an exception from set_terminated must never
    propagate, since this is called from inside an except block and must not mask
    the original exception."""
    client = MagicMock()
    client.set_terminated.side_effect = RuntimeError("tracking server unreachable")
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        mark_run_failed(run_id="run-abc", tracking_uri="sqlite:///tracking.db")
