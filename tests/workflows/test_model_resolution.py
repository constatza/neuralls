"""Tests for strict model_ref resolution helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Self
from unittest.mock import patch

import pytest
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.model_resolution import (
    ModelResolution,
    PreconditionerResolutionResult,
    _resolve_checkpoint_for_run,
    resolve_model_ref,
    resolve_preconditioner_models,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    LoggedModelRefConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
    StandardPreconditionerConfig,
)
from neuralls.platform.tracking.artifact_access import (
    ArtifactLease,
    ArtifactLeaseManager,
    MlflowArtifactLeaseManager,
    NoopArtifactLeaseManager,
)
from neuralls.platform.tracking.checkpoint_selection import find_single_checkpoint
from neuralls.platform.tracking.model_registry import (
    CHECKPOINT_ARTIFACT_PATH_TAG,
    register_logged_model,
)
from tests.workflows.conftest import LoggedNamedCheckpointsRunFactory, LoggedRunFactory


def _tracking_uri(tmp_path: Path) -> str:
    """Build an absolute SQLite URI under the test temp directory."""
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.fixture
def noop_artifact_leases() -> ArtifactLeaseManager:
    """Lease manager for tests that must not resolve MLflow artifacts."""
    return NoopArtifactLeaseManager()


@pytest.fixture
def artifact_leases(mlflow_tracking_uri: str) -> Iterator[ArtifactLeaseManager]:
    """Lease manager backed by the test MLflow store."""
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    with MlflowArtifactLeaseManager(client=client) as leases:
        yield leases


class _CheckpointLeaseManager:
    """Minimal lease manager for checkpoint selection unit tests."""

    def __init__(self, checkpoint_root: Path, fallback_root: Path | None = None) -> None:
        self._checkpoint_root = checkpoint_root
        self._fallback_root = fallback_root or checkpoint_root
        self.file_calls: list[tuple[str, str]] = []
        self.dir_calls: list[tuple[str, str]] = []

    @property
    def tracking_uri(self) -> str | None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def resolve_file(self, run_id: str, artifact_path: str) -> ArtifactLease:
        self.file_calls.append((run_id, artifact_path))
        return ArtifactLease(
            path=self._fallback_root / artifact_path,
            run_id=run_id,
            artifact_path=artifact_path,
            source_uri="file:///artifacts",
            local_copy=False,
        )

    def resolve_dir(self, run_id: str, artifact_path: str) -> ArtifactLease:
        self.dir_calls.append((run_id, artifact_path))
        root = self._checkpoint_root if artifact_path == "checkpoints" else self._fallback_root
        return ArtifactLease(
            path=root,
            run_id=run_id,
            artifact_path=artifact_path,
            source_uri="file:///artifacts",
            local_copy=False,
        )


def test_resolve_preconditioner_models_keeps_non_neural_specs(
    tmp_path: Path,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Non-neural preconditioners pass through unchanged."""
    jacobi = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    resolved = resolve_preconditioner_models(
        specs=[jacobi],
        tracking_uri=_tracking_uri(tmp_path),
        artifact_leases=noop_artifact_leases,
    )
    assert resolved == [jacobi]


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_sets_resolved_checkpoint(
    mock_resolve_model_ref,
    tmp_path: Path,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Neural specs receive resolved_checkpoint_path from resolver."""
    checkpoint = tmp_path / "resolved.ckpt"
    checkpoint.write_text("checkpoint")
    mock_resolve_model_ref.return_value = ModelResolution(
        model_uri="runs:/run-1/model",
        run_id="run-1",
        checkpoint_path=checkpoint,
    )

    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(run_id="run-1"),
    )

    resolved = resolve_preconditioner_models(
        specs=[neural],
        tracking_uri=_tracking_uri(tmp_path),
        artifact_leases=noop_artifact_leases,
    )
    assert len(resolved) == 1
    resolved_neural = resolved[0]
    assert isinstance(resolved_neural, NeuralPreconditionerConfig)
    assert resolved_neural.type == PreconditionerType.NEURAL
    assert resolved_neural.checkpoint_path == checkpoint
    assert resolved_neural.resolved_checkpoint_path == checkpoint


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_keeps_explicit_checkpoint(
    mock_resolve_model_ref,
    tmp_path: Path,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Neural specs with explicit checkpoint_path bypass model_ref resolution."""
    checkpoint = tmp_path / "explicit.ckpt"
    checkpoint.write_text("checkpoint")
    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=checkpoint,
    )

    resolved = resolve_preconditioner_models(
        specs=[neural],
        tracking_uri=_tracking_uri(tmp_path),
        artifact_leases=noop_artifact_leases,
    )

    mock_resolve_model_ref.assert_not_called()
    assert len(resolved) == 1
    resolved_neural = resolved[0]
    assert isinstance(resolved_neural, NeuralPreconditionerConfig)
    assert resolved_neural.checkpoint_path == checkpoint
    assert resolved_neural.resolved_checkpoint_path == checkpoint


def test_resolve_preconditioner_models_requires_model_source(
    tmp_path: Path,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Neural specs without checkpoint_path/model_ref are invalid for resolution."""
    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
    )

    with pytest.raises(ValueError, match="requires either checkpoint_path or model_ref"):
        resolve_preconditioner_models(
            specs=[neural],
            tracking_uri=_tracking_uri(tmp_path),
            artifact_leases=noop_artifact_leases,
        )


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_with_warnings_skips_unresolved_neural(
    mock_resolve_model_ref,
    tmp_path: Path,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Comparison-specific resolution can skip unresolved neural specs."""
    mock_resolve_model_ref.side_effect = ValueError("Registered model 'MissingFFNN' not found")
    jacobi = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    neural = NeuralPreconditionerConfig(
        name="missing-neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name="MissingFFNN", alias="solutions"),
    )

    resolved = resolve_preconditioner_models_with_warnings(
        specs=[jacobi, neural],
        tracking_uri=_tracking_uri(tmp_path),
        artifact_leases=noop_artifact_leases,
        skip_unresolved=True,
    )

    assert isinstance(resolved, PreconditionerResolutionResult)
    assert resolved.specs == [jacobi]
    assert len(resolved.warnings) == 1
    assert "Skipping missing-neural" in resolved.warnings[0]


@patch("neuralls.composition.assignments.model_resolution.MlflowClient")
@patch("neuralls.composition.assignments.model_resolution.search_registered_models")
def test_resolve_model_ref_dataset_placeholder_requires_dataset_alias(
    mock_client_cls,
    mock_search_registered_models,
    tmp_path: Path,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Using alias=@dataset requires a provided dataset alias context."""
    mock_client_cls.return_value = object()
    mock_search_registered_models.return_value = [object()]
    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(
            name="NormScaledLinearFFNN",
            alias="@dataset",
        ),
    )
    with pytest.raises(ValueError, match="requires general\\.data\\.dataset_alias"):
        resolve_model_ref(
            spec=neural,
            tracking_uri=_tracking_uri(tmp_path),
            artifact_leases=noop_artifact_leases,
        )


def test_resolve_model_ref_rejects_mismatched_artifact_lease_store(tmp_path: Path) -> None:
    """Model metadata and artifact leases must come from the same tracking store."""

    class MismatchedLeaseManager(NoopArtifactLeaseManager):
        @property
        def tracking_uri(self) -> str | None:
            return "sqlite:////tmp/other-mlflow.db"

    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(run_id="run-1"),
    )

    with pytest.raises(ValueError, match="must match"):
        resolve_model_ref(
            spec=neural,
            tracking_uri=_tracking_uri(tmp_path),
            artifact_leases=MismatchedLeaseManager(),
        )


@patch("neuralls.composition.assignments.model_resolution.search_registered_models")
@patch("neuralls.composition.assignments.model_resolution.MlflowClient")
def test_resolve_model_ref_normalizes_explicit_at_alias(
    mock_client_cls,
    mock_search_registered_models,
    tmp_path: Path,
) -> None:
    """Registered alias values strip @ prefix before MLflow lookup."""
    mock_search_registered_models.return_value = [object()]
    client = mock_client_cls.return_value
    version = client.get_model_version_by_alias.return_value
    version.run_id = "run-1"
    version.version = "1"
    version.tags = {CHECKPOINT_ARTIFACT_PATH_TAG: "checkpoints/best.ckpt"}
    artifact_leases = _CheckpointLeaseManager(checkpoint_root=tmp_path)

    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(
            name="NormScaledLinearFFNN",
            alias="@solutions",
        ),
    )
    resolve_model_ref(
        spec=neural,
        tracking_uri=_tracking_uri(tmp_path),
        artifact_leases=artifact_leases,
    )

    client.get_model_version_by_alias.assert_called_once_with(
        "NormScaledLinearFFNN",
        "solutions",
    )
    assert artifact_leases.file_calls == [("run-1", "checkpoints/best.ckpt")]


def test_find_single_checkpoint_collapses_nested_duplicate_copy(tmp_path: Path) -> None:
    """Nested duplicate artifact copies should resolve to one canonical checkpoint."""
    root = tmp_path / "run-download"
    duplicate_root = root / "checkpoints"
    root.mkdir(parents=True)
    duplicate_root.mkdir(parents=True)
    primary = root / "model.ckpt"
    duplicate = duplicate_root / "model.ckpt"
    primary.write_bytes(b"same-checkpoint")
    duplicate.write_bytes(b"same-checkpoint")

    resolved = find_single_checkpoint(root)

    assert resolved == primary


def test_find_single_checkpoint_prefers_best_over_last(tmp_path: Path) -> None:
    root = tmp_path / "run-download"
    root.mkdir(parents=True)
    best = root / "best.ckpt"
    best.write_bytes(b"best")
    (root / "last.ckpt").write_bytes(b"last")

    resolved = find_single_checkpoint(root)

    assert resolved == best


def test_find_single_checkpoint_rejects_multiple_best_candidates(tmp_path: Path) -> None:
    root = tmp_path / "run-download"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "best.ckpt").write_bytes(b"best-a")
    (root / "b" / "best.ckpt").write_bytes(b"best-b")
    (root / "last.ckpt").write_bytes(b"last")

    with pytest.raises(ValueError, match="Multiple distinct checkpoints found"):
        find_single_checkpoint(root)


def test_find_single_checkpoint_rejects_interval_checkpoint_with_last(tmp_path: Path) -> None:
    root = tmp_path / "run-download"
    root.mkdir(parents=True)
    (root / "epoch=12-step=500.ckpt").write_bytes(b"interval")
    (root / "last.ckpt").write_bytes(b"last")

    with pytest.raises(ValueError, match="Multiple distinct checkpoints found"):
        find_single_checkpoint(root)


def test_find_single_checkpoint_rejects_distinct_candidates(tmp_path: Path) -> None:
    """Multiple distinct checkpoint files in one run root should fail loudly."""
    root = tmp_path / "run-download"
    root.mkdir(parents=True)
    (root / "first.ckpt").write_bytes(b"first")
    (root / "second.ckpt").write_bytes(b"second")

    with pytest.raises(ValueError, match="Multiple distinct checkpoints found"):
        find_single_checkpoint(root)


def test_find_single_checkpoint_stays_within_provided_root(tmp_path: Path) -> None:
    """Checkpoint discovery must not inspect sibling download roots."""
    target_root = tmp_path / "run-a"
    sibling_root = tmp_path / "run-b"
    target_root.mkdir(parents=True)
    sibling_root.mkdir(parents=True)
    expected = target_root / "model.ckpt"
    expected.write_bytes(b"target")
    (sibling_root / "other.ckpt").write_bytes(b"sibling")

    resolved = find_single_checkpoint(target_root)

    assert resolved == expected


def test_resolve_checkpoint_for_run_collapses_nested_duplicate_artifacts(
    tmp_path: Path,
) -> None:
    """Duplicate artifact trees from one run should resolve to one checkpoint."""
    artifact_root = tmp_path / "artifacts"
    nested = artifact_root / "checkpoints"
    nested.mkdir(parents=True)
    primary = artifact_root / "model.ckpt"
    duplicate = nested / "model.ckpt"
    primary.write_bytes(b"same-checkpoint")
    duplicate.write_bytes(b"same-checkpoint")

    resolved = _resolve_checkpoint_for_run(
        artifact_leases=_CheckpointLeaseManager(checkpoint_root=artifact_root),
        run_id="run-1",
        fallback_artifact_path="model",
    )

    assert resolved == primary


def test_resolve_checkpoint_for_run_raises_on_ambiguous_distinct_artifacts(
    tmp_path: Path,
) -> None:
    """Distinct checkpoint candidates from one run should raise."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True)
    (artifact_root / "first.ckpt").write_bytes(b"first")
    (artifact_root / "second.ckpt").write_bytes(b"second")

    with pytest.raises(ValueError, match="Multiple distinct checkpoints found"):
        _resolve_checkpoint_for_run(
            artifact_leases=_CheckpointLeaseManager(checkpoint_root=artifact_root),
            run_id="run-1",
            fallback_artifact_path="model",
        )


# ---------------------------------------------------------------------------
# _resolve_checkpoint_for_run — primary/fallback candidate ordering
#
# Characterizes which artifact location is consulted, in what order, and which
# exception type survives each stage. The primary location is the canonical
# `checkpoints/` dir; the fallback is the ref's own `artifact_path`.
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_for_run_falls_back_when_primary_dir_holds_no_checkpoint(
    tmp_path: Path,
) -> None:
    """An empty `checkpoints/` dir falls through to the ref's own artifact path."""
    checkpoint_root = tmp_path / "checkpoints"
    fallback_root = tmp_path / "fallback"
    checkpoint_root.mkdir(parents=True)
    fallback_root.mkdir(parents=True)
    expected = fallback_root / "model.ckpt"
    expected.write_bytes(b"fallback")

    leases = _CheckpointLeaseManager(
        checkpoint_root=checkpoint_root,
        fallback_root=fallback_root,
    )
    resolved = _resolve_checkpoint_for_run(
        artifact_leases=leases,
        run_id="run-1",
        fallback_artifact_path="model",
    )

    assert resolved == expected
    assert ("run-1", "checkpoints") in leases.dir_calls
    assert ("run-1", "model") in leases.dir_calls


def test_resolve_checkpoint_for_run_downloads_a_ckpt_suffixed_fallback_as_a_file(
    tmp_path: Path,
) -> None:
    """A fallback artifact path ending in .ckpt is leased as a file, not scanned as a dir."""
    checkpoint_root = tmp_path / "checkpoints"
    fallback_root = tmp_path / "fallback"
    checkpoint_root.mkdir(parents=True)
    fallback_root.mkdir(parents=True)

    leases = _CheckpointLeaseManager(
        checkpoint_root=checkpoint_root,
        fallback_root=fallback_root,
    )
    resolved = _resolve_checkpoint_for_run(
        artifact_leases=leases,
        run_id="run-1",
        fallback_artifact_path="model/best.ckpt",
    )

    assert resolved == fallback_root / "model/best.ckpt"
    assert leases.file_calls == [("run-1", "model/best.ckpt")]


def test_resolve_checkpoint_for_run_raises_file_not_found_when_neither_location_has_one(
    tmp_path: Path,
) -> None:
    """Exhausting both locations reports both of them in one FileNotFoundError."""
    checkpoint_root = tmp_path / "checkpoints"
    fallback_root = tmp_path / "fallback"
    checkpoint_root.mkdir(parents=True)
    fallback_root.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Could not resolve checkpoint artifacts"):
        _resolve_checkpoint_for_run(
            artifact_leases=_CheckpointLeaseManager(
                checkpoint_root=checkpoint_root,
                fallback_root=fallback_root,
            ),
            run_id="run-1",
            fallback_artifact_path="model",
        )


def test_resolve_checkpoint_for_run_propagates_ambiguity_raised_by_the_fallback(
    tmp_path: Path,
) -> None:
    """An ambiguous fallback dir raises ValueError — it is never masked as 'not found'."""
    checkpoint_root = tmp_path / "checkpoints"
    fallback_root = tmp_path / "fallback"
    checkpoint_root.mkdir(parents=True)
    fallback_root.mkdir(parents=True)
    (fallback_root / "first.ckpt").write_bytes(b"first")
    (fallback_root / "second.ckpt").write_bytes(b"second")

    with pytest.raises(ValueError, match="Multiple distinct checkpoints found"):
        _resolve_checkpoint_for_run(
            artifact_leases=_CheckpointLeaseManager(
                checkpoint_root=checkpoint_root,
                fallback_root=fallback_root,
            ),
            run_id="run-1",
            fallback_artifact_path="model",
        )


def test_resolve_registered_ref_latest_picks_highest_version(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """latest=True must select the highest registered version, not run order.

    Registration happens out of chronological run-creation order: the
    highest version number ends up backing the *earliest*-created run, so a
    resolver that accidentally selects "most recently created run" instead
    of "highest version number" fails this test.
    """
    run_a = log_run_with_checkpoint("marker-a")
    run_b = log_run_with_checkpoint("marker-b")
    run_c = log_run_with_checkpoint("marker-c")

    model_name = "LatestVersionModel"
    register_logged_model(
        run_id=run_b, registered_model_name=model_name, tracking_uri=mlflow_tracking_uri
    )  # v1
    register_logged_model(
        run_id=run_c, registered_model_name=model_name, tracking_uri=mlflow_tracking_uri
    )  # v2
    register_logged_model(
        run_id=run_a, registered_model_name=model_name, tracking_uri=mlflow_tracking_uri
    )  # v3

    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name=model_name, latest=True),
    )

    resolution = resolve_model_ref(
        spec=spec,
        tracking_uri=mlflow_tracking_uri,
        artifact_leases=artifact_leases,
    )

    assert resolution.run_id == run_a
    assert resolution.checkpoint_path.read_text() == "marker-a"


def test_resolve_logged_ref_latest_picks_most_recently_started_run(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    log_run_with_checkpoint: LoggedRunFactory,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """latest=True for a logged-model ref selects the most-recently-started run.

    The "new" run's DB record is created before the "old" one, decoupling
    start_time ordering from row-insertion ordering, so the test can't pass
    by accident on insertion order rather than the intended
    ``attributes.start_time DESC`` ordering.
    """
    run_new = log_run_with_checkpoint("marker-new", start_time_ms=5_000)
    run_old = log_run_with_checkpoint("marker-old", start_time_ms=1_000)
    assert run_old != run_new

    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(latest=True, experiment_name=mlflow_experiment),
    )

    resolution = resolve_model_ref(
        spec=spec,
        tracking_uri=mlflow_tracking_uri,
        artifact_leases=artifact_leases,
    )

    assert resolution.run_id == run_new
    assert resolution.checkpoint_path.read_text() == "marker-new"


def test_resolve_preconditioner_models_with_warnings_skips_unresolved_and_continues(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """skip_unresolved=True keeps a batch going and records one warning per failure.

    Without skip_unresolved, the same batch must raise instead.
    """
    run_id = log_run_with_checkpoint("marker-ok")
    resolvable = NeuralPreconditionerConfig(
        name="neural-ok",
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(run_id=run_id),
    )
    unresolvable = NeuralPreconditionerConfig(
        name="neural-missing",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name="DoesNotExistModel", latest=True),
    )

    result = resolve_preconditioner_models_with_warnings(
        specs=[resolvable, unresolvable],
        tracking_uri=mlflow_tracking_uri,
        artifact_leases=artifact_leases,
        skip_unresolved=True,
    )

    assert len(result.specs) == 1
    resolved_spec = result.specs[0]
    assert isinstance(resolved_spec, NeuralPreconditionerConfig)
    assert resolved_spec.name == "neural-ok"
    assert resolved_spec.resolved_checkpoint_path is not None
    assert len(result.warnings) == 1
    assert "neural-missing" in result.warnings[0]
    assert "DoesNotExistModel" in result.warnings[0]

    with pytest.raises(ValueError, match="DoesNotExistModel"):
        resolve_preconditioner_models_with_warnings(
            specs=[resolvable, unresolvable],
            tracking_uri=mlflow_tracking_uri,
            artifact_leases=artifact_leases,
        )


def test_resolve_registered_ref_uses_pinned_checkpoint_without_rescanning(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    log_run_with_named_checkpoints: LoggedNamedCheckpointsRunFactory,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """Resolution downloads the pinned artifact directly, ignoring later ambiguity.

    After registration pins ``checkpoints/best.ckpt``, a second, distinct
    ``best.ckpt`` is added under a nested path so that a rescan (rather than a
    direct pinned download) would raise on ambiguity. Resolution must still
    succeed and return exactly the originally-pinned file, proving it never
    rescans ``checkpoints/``.
    """
    run_id = log_run_with_named_checkpoints(
        {"best.ckpt": "best-content", "last.ckpt": "last-content"}
    )
    model_name = "PinnedNoRescanModel"
    record = register_logged_model(
        run_id=run_id,
        registered_model_name=model_name,
        tracking_uri=mlflow_tracking_uri,
    )

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    ambiguous_best = tmp_path / "nested" / "best.ckpt"
    ambiguous_best.parent.mkdir(parents=True)
    ambiguous_best.write_text("ambiguous-best-content")
    client.log_artifact(run_id, str(ambiguous_best), artifact_path="checkpoints/nested")

    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name=model_name, version=record.version),
    )

    resolution = resolve_model_ref(
        spec=spec,
        tracking_uri=mlflow_tracking_uri,
        artifact_leases=artifact_leases,
    )

    assert resolution.checkpoint_path.name == "best.ckpt"
    assert resolution.checkpoint_path.read_text() == "best-content"


def test_resolve_registered_ref_without_pinned_tag_raises_actionable_error(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """A registered version created before checkpoint pinning cannot be resolved."""
    run_id = log_run_with_checkpoint("marker-legacy")
    model_name = "LegacyUnpinnedModel"
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    client.create_registered_model(model_name)
    client.create_model_version(name=model_name, source=f"runs:/{run_id}/model", run_id=run_id)

    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name=model_name, version=1),
    )

    with pytest.raises(ValueError, match="predates checkpoint pinning"):
        resolve_model_ref(
            spec=spec,
            tracking_uri=mlflow_tracking_uri,
            artifact_leases=artifact_leases,
        )


# ==============================================================================
# CheckpointRefBearing: symmetric resolution across nested checkpoint refs
# ==============================================================================


def test_resolve_preconditioner_models_resolves_neural_pod_coarsening_ref(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """A checkpoint ref nested in AMGPreconditionerConfig.coarsening gets resolved.

    Before the generic CheckpointRefBearing resolver, only the top-level
    `type == "neural"` spec was resolved — this is the gap the new
    NeuralPODCoarseningConfig would otherwise fall into.
    """
    run_id = log_run_with_checkpoint("marker-neural-pod")
    spec = AMGPreconditionerConfig(
        name="neural-pod2g",
        coarsening=NeuralPODCoarseningConfig(
            dataset_dir=tmp_path / "params-dataset",
            input_names=("young_modulus",),
            model_ref=LoggedModelRefConfig(run_id=run_id),
        ),
    )

    resolved = resolve_preconditioner_models(
        specs=[spec],
        tracking_uri=mlflow_tracking_uri,
        artifact_leases=artifact_leases,
    )

    assert len(resolved) == 1
    resolved_spec = resolved[0]
    assert isinstance(resolved_spec, AMGPreconditionerConfig)
    resolved_coarsening = resolved_spec.coarsening
    assert isinstance(resolved_coarsening, NeuralPODCoarseningConfig)
    assert resolved_coarsening.resolved_checkpoint_path is not None
    assert resolved_coarsening.resolved_checkpoint_path.read_text() == "marker-neural-pod"


def test_resolve_preconditioner_models_resolves_neural_amg_prolongation_and_restriction(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    log_run_with_checkpoint: LoggedRunFactory,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """Prolongation and restriction refs resolve independently to distinct checkpoints.

    Distinct source runs prove the two refs aren't accidentally resolved to
    the same download directory (a `name`-only download dirname would
    collide, since both refs share one spec name).
    """
    prolongation_run = log_run_with_checkpoint("marker-prolongation")
    restriction_run = log_run_with_checkpoint("marker-restriction")
    spec = NeuralAMGPreconditionerConfig(
        name="neural-amg",
        prolongation=NeuralTransferConfig(model_ref=LoggedModelRefConfig(run_id=prolongation_run)),
        restriction=NeuralTransferConfig(model_ref=LoggedModelRefConfig(run_id=restriction_run)),
    )

    resolved = resolve_preconditioner_models(
        specs=[spec],
        tracking_uri=mlflow_tracking_uri,
        artifact_leases=artifact_leases,
    )

    assert len(resolved) == 1
    resolved_spec = resolved[0]
    assert isinstance(resolved_spec, NeuralAMGPreconditionerConfig)
    assert resolved_spec.prolongation.resolved_checkpoint_path is not None
    assert resolved_spec.prolongation.resolved_checkpoint_path.read_text() == "marker-prolongation"
    assert resolved_spec.restriction is not None
    assert resolved_spec.restriction.resolved_checkpoint_path is not None
    assert resolved_spec.restriction.resolved_checkpoint_path.read_text() == "marker-restriction"
    assert (
        resolved_spec.prolongation.resolved_checkpoint_path
        != resolved_spec.restriction.resolved_checkpoint_path
    )


def test_resolve_preconditioner_models_with_warnings_labels_failed_nested_ref(
    tmp_path: Path,
    mlflow_tracking_uri: str,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """A failed nested ref's warning names its label, not just the spec name."""
    spec = NeuralAMGPreconditionerConfig(
        name="neural-amg-broken",
        prolongation=NeuralTransferConfig(
            model_ref=RegisteredModelRefConfig(name="DoesNotExistModel", latest=True)
        ),
    )

    result = resolve_preconditioner_models_with_warnings(
        specs=[spec],
        tracking_uri=mlflow_tracking_uri,
        artifact_leases=artifact_leases,
        skip_unresolved=True,
    )

    assert result.specs == []
    assert len(result.warnings) == 1
    assert "neural-amg-broken (prolongation)" in result.warnings[0]
