"""Checkpoint resolution from real artifact directories and a fake MLflow store."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from neuralls.composition.assignments.model_resolution import (
    _resolve_checkpoint_for_run,
    resolve_model_ref,
    resolve_preconditioner_models,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.platform.config.models.checkpoint_refs import (
    ByRunId,
    ByVersion,
    LatestMatching,
    LatestVersion,
    LoggedModelRefConfig,
    RegisteredModelRefConfig,
)
from neuralls.platform.config.models.coarsening import NeuralPODCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PreconditionerType,
)
from neuralls.platform.tracking.artifact_access import (
    ArtifactLeaseManager,
    MlflowArtifactLeaseManager,
)
from neuralls.platform.tracking.services import MlflowTrackingServices
from tests.fakes.checkpoint_lease_manager import CheckpointLeaseManager
from tests.fakes.mlflow_client import FAKE_EXPERIMENT_NAME, FakeMlflowClient


@pytest.fixture
def fake_client(tmp_path: Path) -> FakeMlflowClient:
    """In-memory MLflow store whose run artifacts live under ``tmp_path``."""
    return FakeMlflowClient(artifact_root=tmp_path / "artifacts")


@pytest.fixture
def artifact_leases(fake_client: FakeMlflowClient) -> Iterator[ArtifactLeaseManager]:
    """Real lease manager over the fake store's local artifact directories."""
    with MlflowArtifactLeaseManager(client=fake_client) as leases:
        yield leases


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
        artifact_leases=CheckpointLeaseManager(checkpoint_root=artifact_root),
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

    with pytest.raises(ValueError):
        _resolve_checkpoint_for_run(
            artifact_leases=CheckpointLeaseManager(checkpoint_root=artifact_root),
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

    leases = CheckpointLeaseManager(
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

    leases = CheckpointLeaseManager(
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

    with pytest.raises(FileNotFoundError):
        _resolve_checkpoint_for_run(
            artifact_leases=CheckpointLeaseManager(
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

    with pytest.raises(ValueError):
        _resolve_checkpoint_for_run(
            artifact_leases=CheckpointLeaseManager(
                checkpoint_root=checkpoint_root,
                fallback_root=fallback_root,
            ),
            run_id="run-1",
            fallback_artifact_path="model",
        )


def test_resolve_registered_ref_latest_picks_highest_version(
    fake_client: FakeMlflowClient,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """latest=True must select the highest registered version, not run order.

    Registration happens out of chronological run-creation order: the
    highest version number ends up backing the *earliest*-created run, so a
    resolver that accidentally selects "most recently created run" instead
    of "highest version number" fails this test.
    """
    run_a = fake_client.add_run("marker-a")
    run_b = fake_client.add_run("marker-b")
    run_c = fake_client.add_run("marker-c")

    model_name = "LatestVersionModel"
    for run_id, marker in ((run_b, "b"), (run_c, "c"), (run_a, "a")):
        fake_client.register_model_version(
            model_name, run_id, pinned_path=f"checkpoints/marker-{marker}.ckpt"
        )

    spec = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name=model_name, selector=LatestVersion()),
    )

    resolution = resolve_model_ref(
        spec=spec,
        tracking_uri=fake_client.tracking_uri,
        artifact_leases=artifact_leases,
        tracking=MlflowTrackingServices(client_factory=fake_client),
    )

    assert resolution.run_id == run_a
    assert resolution.checkpoint_path.read_text(encoding="utf-8") == "marker-a"


def test_resolve_logged_ref_latest_picks_most_recently_started_run(
    fake_client: FakeMlflowClient,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """latest=True for a logged-model ref selects the most-recently-started run.

    The "new" run is added before the "old" one, decoupling start_time
    ordering from insertion ordering, so the test can't pass by accident on
    insertion order rather than the intended ``attributes.start_time DESC``
    ordering.
    """
    run_new = fake_client.add_run("marker-new", start_time_ms=5_000)
    run_old = fake_client.add_run("marker-old", start_time_ms=1_000)
    assert run_old != run_new

    spec = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(
            selector=LatestMatching(experiment_name=FAKE_EXPERIMENT_NAME)
        ),
    )

    resolution = resolve_model_ref(
        spec=spec,
        tracking_uri=fake_client.tracking_uri,
        artifact_leases=artifact_leases,
        tracking=MlflowTrackingServices(client_factory=fake_client),
    )

    assert resolution.run_id == run_new
    assert resolution.checkpoint_path.read_text(encoding="utf-8") == "marker-new"


def test_resolve_preconditioner_models_with_warnings_skips_unresolved_and_continues(
    fake_client: FakeMlflowClient,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """skip_unresolved=True keeps a batch going and records one warning per failure.

    Without skip_unresolved, the same batch must raise instead.
    """
    run_id = fake_client.add_run("marker-ok")
    resolvable = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(selector=ByRunId(run_id=run_id)),
    )
    unresolvable = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name="DoesNotExistModel", selector=LatestVersion()),
    )

    result = resolve_preconditioner_models_with_warnings(
        specs=[resolvable, unresolvable],
        tracking_uri=fake_client.tracking_uri,
        artifact_leases=artifact_leases,
        tracking=MlflowTrackingServices(client_factory=fake_client),
        skip_unresolved=True,
    )

    assert len(result.specs) == 1
    resolved_spec = result.specs[0]
    assert isinstance(resolved_spec, NeuralPreconditionerConfig)
    assert resolved_spec.name == resolvable.name
    assert resolved_spec.resolved_checkpoint_path is not None
    assert len(result.warnings) == 1

    with pytest.raises(ValueError):
        resolve_preconditioner_models_with_warnings(
            specs=[resolvable, unresolvable],
            tracking_uri=fake_client.tracking_uri,
            artifact_leases=artifact_leases,
            tracking=MlflowTrackingServices(client_factory=fake_client),
        )


def test_resolve_registered_ref_uses_pinned_checkpoint_without_rescanning(
    fake_client: FakeMlflowClient,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """Resolution downloads the pinned artifact directly, ignoring later ambiguity.

    After registration pins ``checkpoints/best.ckpt``, a second, distinct
    ``best.ckpt`` is added under a nested path so that a rescan (rather than a
    direct pinned download) would raise on ambiguity. Resolution must still
    succeed and return exactly the originally-pinned file, proving it never
    rescans ``checkpoints/``.
    """
    run_id = fake_client.add_run(
        "pinned", checkpoints={"best.ckpt": "best-content", "last.ckpt": "last-content"}
    )
    model_name = "PinnedNoRescanModel"
    version = fake_client.register_model_version(
        model_name, run_id, pinned_path="checkpoints/best.ckpt"
    )
    fake_client.add_run_artifact(run_id, "checkpoints/nested/best.ckpt", "ambiguous-best-content")

    spec = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name=model_name, selector=ByVersion(version=version)),
    )

    resolution = resolve_model_ref(
        spec=spec,
        tracking_uri=fake_client.tracking_uri,
        artifact_leases=artifact_leases,
        tracking=MlflowTrackingServices(client_factory=fake_client),
    )

    assert resolution.checkpoint_path.name == "best.ckpt"
    assert resolution.checkpoint_path.read_text(encoding="utf-8") == "best-content"


def test_resolve_registered_ref_without_pinned_tag_is_rejected(
    fake_client: FakeMlflowClient,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """A registered version created before checkpoint pinning cannot be resolved."""
    run_id = fake_client.add_run("marker-legacy")
    model_name = "LegacyUnpinnedModel"
    fake_client.register_model_version(model_name, run_id, pinned_path=None)

    spec = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name=model_name, selector=ByVersion(version=1)),
    )

    with pytest.raises(ValueError):
        resolve_model_ref(
            spec=spec,
            tracking_uri=fake_client.tracking_uri,
            artifact_leases=artifact_leases,
            tracking=MlflowTrackingServices(client_factory=fake_client),
        )


# ==============================================================================
# CheckpointRefBearing: symmetric resolution across nested checkpoint refs
# ==============================================================================


def test_resolve_preconditioner_models_resolves_neural_pod_coarsening_ref(
    tmp_path: Path,
    fake_client: FakeMlflowClient,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """A checkpoint ref nested in AMGPreconditionerConfig.coarsening gets resolved.

    Before the generic CheckpointRefBearing resolver, only the top-level
    `type == "neural"` spec was resolved — this is the gap the new
    NeuralPODCoarseningConfig would otherwise fall into.
    """
    run_id = fake_client.add_run("marker-neural-pod")
    spec = AMGPreconditionerConfig(
        coarsening=NeuralPODCoarseningConfig(
            dataset_dir=tmp_path / "params-dataset",
            input_names=("young_modulus",),
            model_ref=LoggedModelRefConfig(selector=ByRunId(run_id=run_id)),
        ),
    )

    resolved = resolve_preconditioner_models(
        specs=[spec],
        tracking_uri=fake_client.tracking_uri,
        artifact_leases=artifact_leases,
        tracking=MlflowTrackingServices(client_factory=fake_client),
    )

    assert len(resolved) == 1
    resolved_spec = resolved[0]
    assert isinstance(resolved_spec, AMGPreconditionerConfig)
    resolved_coarsening = resolved_spec.coarsening
    assert isinstance(resolved_coarsening, NeuralPODCoarseningConfig)
    assert resolved_coarsening.resolved_checkpoint_path is not None
    assert (
        resolved_coarsening.resolved_checkpoint_path.read_text(encoding="utf-8")
        == "marker-neural-pod"
    )


def test_resolve_preconditioner_models_resolves_neural_amg_prolongation_and_restriction(
    fake_client: FakeMlflowClient,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """Prolongation and restriction refs resolve independently to distinct checkpoints.

    Distinct source runs prove the two refs aren't accidentally resolved to
    the same download directory (a `name`-only download dirname would
    collide, since both refs share one spec name).
    """
    prolongation_run = fake_client.add_run("marker-prolongation")
    restriction_run = fake_client.add_run("marker-restriction")
    spec = NeuralAMGPreconditionerConfig(
        prolongation=NeuralTransferConfig(
            model_ref=LoggedModelRefConfig(selector=ByRunId(run_id=prolongation_run))
        ),
        restriction=NeuralTransferConfig(
            model_ref=LoggedModelRefConfig(selector=ByRunId(run_id=restriction_run))
        ),
    )

    resolved = resolve_preconditioner_models(
        specs=[spec],
        tracking_uri=fake_client.tracking_uri,
        artifact_leases=artifact_leases,
        tracking=MlflowTrackingServices(client_factory=fake_client),
    )

    assert len(resolved) == 1
    resolved_spec = resolved[0]
    assert isinstance(resolved_spec, NeuralAMGPreconditionerConfig)
    assert resolved_spec.restriction is not None
    assert resolved_spec.restriction.resolved_checkpoint_path is not None
    assert resolved_spec.prolongation.resolved_checkpoint_path is not None
    assert (
        resolved_spec.prolongation.resolved_checkpoint_path.read_text(encoding="utf-8")
        == "marker-prolongation"
    )
    assert (
        resolved_spec.restriction.resolved_checkpoint_path.read_text(encoding="utf-8")
        == "marker-restriction"
    )
    assert (
        resolved_spec.prolongation.resolved_checkpoint_path
        != resolved_spec.restriction.resolved_checkpoint_path
    )


def test_resolve_preconditioner_models_with_warnings_skips_failed_nested_ref(
    fake_client: FakeMlflowClient,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    """A failed nested ref's warning names its label, not just the spec name."""
    spec = NeuralAMGPreconditionerConfig(
        prolongation=NeuralTransferConfig(
            model_ref=RegisteredModelRefConfig(name="DoesNotExistModel", selector=LatestVersion())
        ),
    )

    result = resolve_preconditioner_models_with_warnings(
        specs=[spec],
        tracking_uri=fake_client.tracking_uri,
        artifact_leases=artifact_leases,
        tracking=MlflowTrackingServices(client_factory=fake_client),
        skip_unresolved=True,
    )

    assert result.specs == []
    assert len(result.warnings) == 1
