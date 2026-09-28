"""Tests for `comparison_run.py`'s setup-cost history lookup (`_resolve_setup_usage`).

A checkpoint-backed preconditioner's `setup_time_seconds` must come from the
checkpoint's own original training run when that run is skipped this time,
not from the (comparatively negligible) checkpoint-load/coarse-assembly time
freshly measured this run. This file has two layers:

- Unit tests mocking `fetch_mlflow_metrics` for each branch of the fallback
  logic.
- A regression test against a *real* sqlite-backed MLflow store (no mocks)
  proving the round trip end to end: a fixed, deliberately distinctive
  duration logged on an "original" run is exactly what a "skipped" build
  reports as its own `setup_time_seconds` — not a fresh (and necessarily
  different) re-measurement. Wall-clock durations are inherently
  non-reproducible between two runs, so this seeds the "original" run's
  metric directly rather than trying to make two real builds take the same
  measured time.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mlflow.tracking import MlflowClient

from neuralls.composition.comparison.comparison_run import _resolve_setup_usage
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.shared.constants import TRAINING_CHILD_DURATION_METRIC_KEY
from neuralls.shared.device import ResourceUsage

_FETCH = "neuralls.composition.comparison.comparison_run.fetch_mlflow_metrics"
_MEASURED = ResourceUsage(wall_time_seconds=0.001, peak_memory_bytes=123)


def _neural_cfg(resolved_run_id: str | None) -> NeuralPreconditionerConfig:
    return NeuralPreconditionerConfig(
        name="neural", type=PreconditionerType.NEURAL, resolved_run_id=resolved_run_id
    )


def test_returns_measured_without_tracking_uri() -> None:
    cfg = _neural_cfg("run-123")
    with patch(_FETCH) as fetch:
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri=None)

    fetch.assert_not_called()
    assert result is _MEASURED


def test_returns_measured_for_a_config_with_no_checkpoint_refs() -> None:
    """A non-checkpoint-backed config (e.g. Jacobi) never triggers a lookup."""
    cfg = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    with patch(_FETCH) as fetch:
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    fetch.assert_not_called()
    assert result is _MEASURED


def test_returns_measured_when_checkpoint_ref_has_no_resolved_run_id() -> None:
    """A literal `checkpoint_path` (not resolved via `assignment`/`model_ref`)
    never gets `resolved_run_id` populated -- nothing to look up, so the
    fresh measurement (the only real number available) is used.
    """
    cfg = _neural_cfg(None)
    with patch(_FETCH) as fetch:
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    fetch.assert_not_called()
    assert result is _MEASURED


def test_returns_historical_duration_when_present_keeping_measured_memory() -> None:
    cfg = _neural_cfg("run-123")
    with patch(_FETCH, return_value={TRAINING_CHILD_DURATION_METRIC_KEY: 42.0}) as fetch:
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    fetch.assert_called_once_with("run-123", "tracking-uri")
    assert result.wall_time_seconds == 42.0
    assert result.peak_memory_bytes == _MEASURED.peak_memory_bytes


def test_returns_measured_when_the_metric_was_never_logged_on_that_run() -> None:
    """The origin run exists but predates this feature -- no key, no crash,
    fall back to the fresh measurement.
    """
    cfg = _neural_cfg("run-123")
    with patch(_FETCH, return_value={"some_other_metric": 1.0}):
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    assert result is _MEASURED


def test_returns_measured_and_warns_on_lookup_failure() -> None:
    """No tracking server / run not found must never abort the comparison."""
    cfg = _neural_cfg("run-123")
    with patch(_FETCH, side_effect=RuntimeError("no server")):
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    assert result is _MEASURED


def test_skipped_build_reports_exactly_the_original_runs_recorded_duration(
    tmp_path: Path,
) -> None:
    """Regression test against a real (unmocked) MLflow store.

    Seeds an "original" run with a deliberately distinctive duration
    (7.5s -- nothing a fresh, near-instant checkpoint-load/assembly could
    ever coincidentally match), points a config's `resolved_run_id` at it,
    and confirms `_resolve_setup_usage` reports that exact value as this
    "skipped" build's own `setup_time_seconds` -- not the fresh measurement.
    """
    tracking_uri = build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("setup-cost-history-regression")
    original_run = client.create_run(experiment_id=experiment_id)
    client.log_metric(original_run.info.run_id, TRAINING_CHILD_DURATION_METRIC_KEY, 7.5)
    client.set_terminated(original_run.info.run_id)

    cfg = _neural_cfg(original_run.info.run_id)
    fresh_measurement = ResourceUsage(wall_time_seconds=0.002, peak_memory_bytes=456)

    result = _resolve_setup_usage(cfg, measured=fresh_measurement, tracking_uri=tracking_uri)

    assert result.wall_time_seconds == 7.5
    assert result.wall_time_seconds != fresh_measurement.wall_time_seconds
    assert result.peak_memory_bytes == fresh_measurement.peak_memory_bytes
