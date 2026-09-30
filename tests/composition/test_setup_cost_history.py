"""Tests for `comparison_run.py`'s setup-cost history lookup (`_resolve_setup_usage`).

A checkpoint-backed preconditioner's setup wall time must come from the
checkpoint's own original training run when that run is skipped this time,
not from the (comparatively negligible) checkpoint-load/coarse-assembly time
freshly measured this run. This file has two layers:

- Unit tests mocking `fetch_mlflow_metrics` for each branch of the fallback
  logic, each asserting both the numeric wall time and the resulting
  `CostProvenance` -- the two facts a `StageCost` couples so a silent
  lookup failure can never render identically to a genuine near-zero cost.
- A regression test against a *real* sqlite-backed MLflow store (no mocks)
  proving the round trip end to end: a fixed, deliberately distinctive
  duration logged on an "original" run is exactly what a "skipped" build
  reports as its own setup wall time -- not a fresh (and necessarily
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
from neuralls.shared.types import CostProvenance

_FETCH = "neuralls.composition.comparison.comparison_run.fetch_mlflow_metrics"
_MEASURED = ResourceUsage(wall_time_seconds=0.001, peak_memory_bytes=123)


def _neural_cfg(resolved_run_id: str | None) -> NeuralPreconditionerConfig:
    return NeuralPreconditionerConfig(
        name="neural", type=PreconditionerType.NEURAL, resolved_run_id=resolved_run_id
    )


def test_returns_unavailable_provenance_without_tracking_uri() -> None:
    """A checkpoint-backed config with no tracking URI to query is a
    silent-lookup-skip case, not a real measurement -- UNAVAILABLE, not
    MEASURED, even though the numeric value is the fresh measurement.
    """
    cfg = _neural_cfg("run-123")
    with patch(_FETCH) as fetch:
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri=None)

    fetch.assert_not_called()
    assert result.wall_time_seconds == _MEASURED.wall_time_seconds
    assert result.peak_memory_bytes == _MEASURED.peak_memory_bytes
    assert result.provenance is CostProvenance.UNAVAILABLE


def test_returns_measured_for_a_config_with_no_checkpoint_refs() -> None:
    """A non-checkpoint-backed config (e.g. Jacobi) never triggers a lookup,
    and the fresh measurement really is the real cost -- MEASURED.
    """
    cfg = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    with patch(_FETCH) as fetch:
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    fetch.assert_not_called()
    assert result.wall_time_seconds == _MEASURED.wall_time_seconds
    assert result.peak_memory_bytes == _MEASURED.peak_memory_bytes
    assert result.provenance is CostProvenance.MEASURED


def test_returns_unavailable_provenance_when_checkpoint_ref_has_no_resolved_run_id() -> None:
    """A literal `checkpoint_path` (not resolved via `assignment`/`model_ref`)
    never gets `resolved_run_id` populated -- nothing to look up, so the
    fresh measurement is used, but it's still an UNAVAILABLE historical
    charge-back, not a genuine measurement of the real build cost.
    """
    cfg = _neural_cfg(None)
    with patch(_FETCH) as fetch:
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    fetch.assert_not_called()
    assert result.wall_time_seconds == _MEASURED.wall_time_seconds
    assert result.peak_memory_bytes == _MEASURED.peak_memory_bytes
    assert result.provenance is CostProvenance.UNAVAILABLE


def test_returns_historical_duration_when_present_keeping_measured_memory() -> None:
    cfg = _neural_cfg("run-123")
    with patch(_FETCH, return_value={TRAINING_CHILD_DURATION_METRIC_KEY: 42.0}) as fetch:
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    fetch.assert_called_once_with("run-123", "tracking-uri")
    assert result.wall_time_seconds == 42.0
    assert result.peak_memory_bytes == _MEASURED.peak_memory_bytes
    assert result.provenance is CostProvenance.HISTORICAL


def test_returns_unavailable_provenance_when_the_metric_was_never_logged_on_that_run() -> None:
    """The origin run exists but predates this feature -- no key, no crash,
    fall back to the fresh measurement, tagged UNAVAILABLE since it isn't
    a genuine measurement of the real (already-paid) build cost.
    """
    cfg = _neural_cfg("run-123")
    with patch(_FETCH, return_value={"some_other_metric": 1.0}):
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    assert result.wall_time_seconds == _MEASURED.wall_time_seconds
    assert result.peak_memory_bytes == _MEASURED.peak_memory_bytes
    assert result.provenance is CostProvenance.UNAVAILABLE


def test_returns_unavailable_provenance_and_warns_on_lookup_failure() -> None:
    """No tracking server / run not found must never abort the comparison,
    and must never be mistaken for a genuine measurement."""
    cfg = _neural_cfg("run-123")
    with patch(_FETCH, side_effect=RuntimeError("no server")):
        result = _resolve_setup_usage(cfg, measured=_MEASURED, tracking_uri="tracking-uri")

    assert result.wall_time_seconds == _MEASURED.wall_time_seconds
    assert result.peak_memory_bytes == _MEASURED.peak_memory_bytes
    assert result.provenance is CostProvenance.UNAVAILABLE


def test_measured_and_unavailable_are_distinguishable_even_with_identical_wall_time() -> None:
    """The direct regression test for the original bug: a genuinely fast
    build (MEASURED) and a silently-failed historical lookup falling back
    to the same fresh measurement (UNAVAILABLE) must produce the *same*
    numeric wall time but *different* provenance -- so a reader can always
    tell them apart, unlike the original bare-float shape.
    """
    plain_cfg = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    with patch(_FETCH) as fetch:
        measured_result = _resolve_setup_usage(
            plain_cfg, measured=_MEASURED, tracking_uri="tracking-uri"
        )
    fetch.assert_not_called()

    checkpoint_cfg = _neural_cfg(None)
    with patch(_FETCH) as fetch:
        unavailable_result = _resolve_setup_usage(
            checkpoint_cfg, measured=_MEASURED, tracking_uri="tracking-uri"
        )
    fetch.assert_not_called()

    assert measured_result.wall_time_seconds == unavailable_result.wall_time_seconds
    assert measured_result.provenance is CostProvenance.MEASURED
    assert unavailable_result.provenance is CostProvenance.UNAVAILABLE
    assert measured_result.provenance != unavailable_result.provenance


def test_skipped_build_reports_exactly_the_original_runs_recorded_duration(
    tmp_path: Path,
) -> None:
    """Regression test against a real (unmocked) MLflow store.

    Seeds an "original" run with a deliberately distinctive duration
    (7.5s -- nothing a fresh, near-instant checkpoint-load/assembly could
    ever coincidentally match), points a config's `resolved_run_id` at it,
    and confirms `_resolve_setup_usage` reports that exact value as this
    "skipped" build's own setup wall time, tagged HISTORICAL -- not the
    fresh measurement.
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
    assert result.provenance is CostProvenance.HISTORICAL


def test_forced_retrain_reports_the_new_historical_duration_not_the_stale_one(
    tmp_path: Path,
) -> None:
    """Idempotency/robustness regression: a `--force-train`-style second
    training run for the same assignment must never leave a stale
    historical cost around. Simulates: first training run logs D1 -> setup
    cost is HISTORICAL/D1; a forced retrain logs a different D2 onto a NEW
    run id, and the config's resolved run id is updated to point at it (as
    the real resolution flow does after a forced retrain) -> setup cost is
    now HISTORICAL/D2, not D1.
    """
    tracking_uri = build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("forced-retrain-regression")

    first_run = client.create_run(experiment_id=experiment_id)
    client.log_metric(first_run.info.run_id, TRAINING_CHILD_DURATION_METRIC_KEY, 7.5)
    client.set_terminated(first_run.info.run_id)

    measured = ResourceUsage(wall_time_seconds=0.002, peak_memory_bytes=456)
    cfg_first = _neural_cfg(first_run.info.run_id)
    first_result = _resolve_setup_usage(cfg_first, measured=measured, tracking_uri=tracking_uri)
    assert first_result.provenance is CostProvenance.HISTORICAL
    assert first_result.wall_time_seconds == 7.5

    second_run = client.create_run(experiment_id=experiment_id)
    client.log_metric(second_run.info.run_id, TRAINING_CHILD_DURATION_METRIC_KEY, 3.25)
    client.set_terminated(second_run.info.run_id)

    cfg_second = _neural_cfg(second_run.info.run_id)
    second_result = _resolve_setup_usage(cfg_second, measured=measured, tracking_uri=tracking_uri)

    assert second_result.provenance is CostProvenance.HISTORICAL
    assert second_result.wall_time_seconds == 3.25
    assert second_result.wall_time_seconds != first_result.wall_time_seconds
