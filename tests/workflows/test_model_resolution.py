"""Tests for strict model_ref resolution helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Self
from unittest.mock import MagicMock, patch

import pytest

from neuralls.composition.assignments.model_resolution import (
    ModelResolution,
    PreconditionerResolutionResult,
    _resolve_checkpoint_for_run,
    resolve_model_ref,
    resolve_preconditioner_models,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.platform.config.models.checkpoint_refs import (
    ByAlias,
    ByRunId,
    ByVersion,
    LatestMatching,
    LatestVersion,
    LoggedModelRefConfig,
    NeuralCheckpointRef,
    RegisteredModelRefConfig,
)
from neuralls.platform.config.models.coarsening import (
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PreconditionerType,
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
)
from neuralls.platform.tracking.services import MlflowTrackingServices
from tests.fakes.mlflow_client import FAKE_EXPERIMENT_NAME, FakeMlflowClient

pytestmark = pytest.mark.deferred_io("mlflow-run-tracking")

# Never opened: every test patches the MLflow-facing collaborators.
TRACKING_URI = "sqlite:////workspace/mlflow.db"


@pytest.fixture
def noop_artifact_leases() -> ArtifactLeaseManager:
    """Lease manager for tests that must not resolve MLflow artifacts."""
    return NoopArtifactLeaseManager()


@pytest.fixture
def fake_client(tmp_path: Path) -> FakeMlflowClient:
    """In-memory MLflow store whose run artifacts live under ``tmp_path``."""
    return FakeMlflowClient(artifact_root=tmp_path / "artifacts")


@pytest.fixture
def artifact_leases(fake_client: FakeMlflowClient) -> Iterator[ArtifactLeaseManager]:
    """Real lease manager over the fake store's local artifact directories."""
    with MlflowArtifactLeaseManager(client=fake_client) as leases:
        yield leases


@pytest.fixture
def pod_specs_across_datasets(tmp_path: Path) -> list[AMGPreconditionerConfig]:
    """Four uniquely keyed POD-2G cases spanning two datasets and two ranks."""
    cases = (
        ("pod-cg-r10", "cg", 10),
        ("pod-cg-r100", "cg", 100),
        ("pod-smoother-r10", "smoother", 10),
        ("pod-smoother-r100", "smoother", 100),
    )
    return [
        AMGPreconditionerConfig(
            coarsening=PODCoarseningConfig(
                dataset_dir=tmp_path / dataset,
                rank=rank,
                assignment=name,
                model_ref=LoggedModelRefConfig(selector=ByRunId(run_id=name)),
            ),
        )
        for name, dataset, rank in cases
    ]


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
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Non-neural preconditioners pass through unchanged."""
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    resolved = resolve_preconditioner_models(
        specs=[jacobi],
        tracking_uri=TRACKING_URI,
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
    checkpoint.write_text("checkpoint", encoding="utf-8")
    mock_resolve_model_ref.return_value = ModelResolution(
        model_uri="runs:/run-1/model",
        run_id="run-1",
        checkpoint_path=checkpoint,
    )

    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(selector=ByRunId(run_id="run-1")),
    )

    resolved = resolve_preconditioner_models(
        specs=[neural],
        tracking_uri=TRACKING_URI,
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
    checkpoint.write_text("checkpoint", encoding="utf-8")
    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        checkpoint_path=checkpoint,
    )

    resolved = resolve_preconditioner_models(
        specs=[neural],
        tracking_uri=TRACKING_URI,
        artifact_leases=noop_artifact_leases,
    )

    mock_resolve_model_ref.assert_not_called()
    assert len(resolved) == 1
    resolved_neural = resolved[0]
    assert isinstance(resolved_neural, NeuralPreconditionerConfig)
    assert resolved_neural.checkpoint_path == checkpoint
    assert resolved_neural.resolved_checkpoint_path == checkpoint


def test_resolve_preconditioner_models_requires_model_source(
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Neural specs without checkpoint_path/model_ref are invalid for resolution."""
    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
    )

    with pytest.raises(ValueError):
        resolve_preconditioner_models(
            specs=[neural],
            tracking_uri=TRACKING_URI,
            artifact_leases=noop_artifact_leases,
        )


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_with_warnings_skips_unresolved_neural(
    mock_resolve_model_ref,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Comparison-specific resolution can skip unresolved neural specs."""
    mock_resolve_model_ref.side_effect = ValueError("Registered model 'MissingFFNN' not found")
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name="MissingFFNN", selector=ByAlias(alias="solutions")),
    )

    resolved = resolve_preconditioner_models_with_warnings(
        specs=[jacobi, neural],
        tracking_uri=TRACKING_URI,
        artifact_leases=noop_artifact_leases,
        skip_unresolved=True,
    )

    assert isinstance(resolved, PreconditionerResolutionResult)
    assert resolved.specs == [jacobi]
    assert len(resolved.warnings) == 1


def test_resolve_model_ref_dataset_placeholder_requires_dataset_alias(
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Using alias=@dataset requires a provided dataset alias context."""
    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(
            name="NormScaledLinearFFNN",
            selector=ByAlias(alias="@dataset"),
        ),
    )
    with pytest.raises(ValueError):
        resolve_model_ref(
            spec=neural,
            tracking_uri=TRACKING_URI,
            artifact_leases=noop_artifact_leases,
            tracking=MlflowTrackingServices(client_factory=MagicMock()),
        )


def test_resolve_model_ref_rejects_mismatched_artifact_lease_store() -> None:
    """Model metadata and artifact leases must come from the same tracking store."""

    class MismatchedLeaseManager(NoopArtifactLeaseManager):
        @property
        def tracking_uri(self) -> str | None:
            return "sqlite:////tmp/other-mlflow.db"

    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(selector=ByRunId(run_id="run-1")),
    )

    with pytest.raises(ValueError):
        resolve_model_ref(
            spec=neural,
            tracking_uri=TRACKING_URI,
            artifact_leases=MismatchedLeaseManager(),
        )


def test_resolve_model_ref_normalizes_explicit_at_alias(tmp_path: Path) -> None:
    """Registered alias values strip @ prefix before MLflow lookup."""
    client = MagicMock()
    version = client.get_model_version_by_alias.return_value
    version.run_id = "run-1"
    version.version = "1"
    version.tags = {CHECKPOINT_ARTIFACT_PATH_TAG: "checkpoints/best.ckpt"}
    artifact_leases = _CheckpointLeaseManager(checkpoint_root=tmp_path)

    neural = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(
            name="NormScaledLinearFFNN",
            selector=ByAlias(alias="@solutions"),
        ),
    )
    resolve_model_ref(
        spec=neural,
        tracking_uri=TRACKING_URI,
        artifact_leases=artifact_leases,
        tracking=MlflowTrackingServices(client_factory=MagicMock(return_value=client)),
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

    with pytest.raises(ValueError):
        find_single_checkpoint(root)


def test_find_single_checkpoint_rejects_interval_checkpoint_with_last(tmp_path: Path) -> None:
    root = tmp_path / "run-download"
    root.mkdir(parents=True)
    (root / "epoch=12-step=500.ckpt").write_bytes(b"interval")
    (root / "last.ckpt").write_bytes(b"last")

    with pytest.raises(ValueError):
        find_single_checkpoint(root)


def test_find_single_checkpoint_rejects_distinct_candidates(tmp_path: Path) -> None:
    """Multiple distinct checkpoint files in one run root should fail loudly."""
    root = tmp_path / "run-download"
    root.mkdir(parents=True)
    (root / "first.ckpt").write_bytes(b"first")
    (root / "second.ckpt").write_bytes(b"second")

    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
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

    with pytest.raises(FileNotFoundError):
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

    with pytest.raises(ValueError):
        _resolve_checkpoint_for_run(
            artifact_leases=_CheckpointLeaseManager(
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


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_skipping_one_pod_case_preserves_each_survivors_dataset_rank_and_checkpoint(
    mock_resolve_model_ref,
    pod_specs_across_datasets: list[AMGPreconditionerConfig],
    tmp_path: Path,
    noop_artifact_leases: ArtifactLeaseManager,
) -> None:
    """Resolution gaps never shift a same-family checkpoint onto a neighboring case."""

    def resolve_by_assignment(*, spec: NeuralCheckpointRef, **_kwargs: object) -> ModelResolution:
        assignment = spec.assignment
        if assignment == "pod-smoother-r10":
            raise ValueError("intentionally missing")
        checkpoint = tmp_path / f"{assignment}.ckpt"
        return ModelResolution(
            model_uri=f"runs:/{assignment}/model",
            run_id=str(assignment),
            checkpoint_path=checkpoint,
        )

    mock_resolve_model_ref.side_effect = resolve_by_assignment

    result = resolve_preconditioner_models_with_warnings(
        specs=list(pod_specs_across_datasets),
        tracking_uri=TRACKING_URI,
        artifact_leases=noop_artifact_leases,
        skip_unresolved=True,
    )

    actual = []
    for spec in result.specs:
        assert isinstance(spec, AMGPreconditionerConfig)
        assert isinstance(spec.coarsening, PODCoarseningConfig)
        checkpoint = spec.coarsening.resolved_checkpoint_path
        assert checkpoint is not None
        actual.append(
            (
                spec.coarsening.assignment,
                spec.coarsening.dataset_dir.name,
                spec.coarsening.rank,
                checkpoint.name,
            )
        )

    assert actual == [
        ("pod-cg-r10", "cg", 10, "pod-cg-r10.ckpt"),
        ("pod-cg-r100", "cg", 100, "pod-cg-r100.ckpt"),
        ("pod-smoother-r100", "smoother", 100, "pod-smoother-r100.ckpt"),
    ]
    assert len(result.warnings) == 1


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
