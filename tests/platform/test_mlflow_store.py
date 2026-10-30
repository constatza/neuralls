"""`MlflowIdentityStore`: exact-key, experiment-scoped, deterministic run lookup."""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import tomli_w
from mlflow.tracking import MlflowClient
from tests.presentation_support import PreparedMethodFactory
from tests.provenance_support import provenance_for
from tests.recipe_support import GAUSSIAN

from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    canonical_method_config,
)
from neuralls.composition.tracking.run_specs import build_benchmark_method_run_spec
from neuralls.domain.identity import IdentityTag, Reused, StageIdentity
from neuralls.domain.presentation.methods import ExecutedMethod
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SolveAttempt,
    SolveConverged,
    SystemTag,
)
from neuralls.domain.solver.reporting.aggregation import summarize_method_workload
from neuralls.domain.solver.reporting.event_frames import validate_decoded_solve_frame
from neuralls.platform.config.models.preconditioner import NeuralPreconditionerConfig
from neuralls.platform.reporting.benchmark_events import (
    METHOD_ARTIFACT_DIR,
    MethodBenchmarkArtifacts,
    write_method_benchmark_artifacts,
)
from neuralls.platform.tracking.mlflow_store import (
    MlflowIdentityStore,
    _validate_method_artifacts,
    find_reusable_benchmark_method,
)
from neuralls.shared.digest import Json, canonical_digest
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, MatrixId, PreconditionerName, SetupEventId, SystemId

EXPERIMENT = "Train-Case"
CONFIG_DIGEST = "sha256:" + "a" * 64

type RunFactory = Callable[..., str]


@pytest.fixture
def client(mlflow_tracking_uri: str) -> MlflowClient:
    """Client bound to the isolated store."""
    return MlflowClient(tracking_uri=mlflow_tracking_uri)


@pytest.fixture
def identity() -> StageIdentity:
    """Identity under test."""
    return StageIdentity.build(
        "training", {"dataset": canonical_digest("d"), "job": canonical_digest("j")}
    )


@pytest.fixture
def other_identity() -> StageIdentity:
    """Identity differing in one component."""
    return StageIdentity.build(
        "training", {"dataset": canonical_digest("d"), "job": canonical_digest("J2")}
    )


@pytest.fixture
def make_run(client: MlflowClient) -> RunFactory:
    """Create a run with explicit start time, status and tags in a named experiment."""

    def _make(
        identity: StageIdentity,
        *,
        start: int,
        status: str = "FINISHED",
        experiment: str = EXPERIMENT,
    ) -> str:
        found = client.get_experiment_by_name(experiment)
        experiment_id = found.experiment_id if found else client.create_experiment(experiment)
        run = client.create_run(experiment_id, start_time=start, tags=identity.tags())
        client.set_terminated(run.info.run_id, status=status, end_time=start + 1)
        return run.info.run_id

    return _make


@pytest.fixture
def store(mlflow_tracking_uri: str) -> MlflowIdentityStore:
    """Store for the training experiment, no checkpoint requirement."""
    return MlflowIdentityStore(tracking_uri=mlflow_tracking_uri, experiment=EXPERIMENT)


def test_missing_experiment_is_a_miss(store: MlflowIdentityStore, identity: StageIdentity) -> None:
    """No experiment means nothing to reuse."""
    assert store.find(identity) is None


def test_newest_finished_run_with_the_exact_key_wins(
    store: MlflowIdentityStore, identity: StageIdentity, make_run: RunFactory
) -> None:
    """Among equal-key runs the newest wins."""
    make_run(identity, start=1000)
    newest = make_run(identity, start=2000)
    assert store.find(identity) == Reused(run_id=newest)


def test_other_key_failed_and_other_experiment_runs_never_match(
    store: MlflowIdentityStore,
    identity: StageIdentity,
    other_identity: StageIdentity,
    make_run: RunFactory,
) -> None:
    """Different key, non-FINISHED status and other experiments are all ignored."""
    good = make_run(identity, start=1000)
    make_run(other_identity, start=5000)
    make_run(identity, start=6000, status="FAILED")
    make_run(identity, start=7000, experiment="Other")
    assert store.find(identity) == Reused(run_id=good)


def test_run_without_checkpoint_falls_back_to_older_run(
    mlflow_tracking_uri: str, identity: StageIdentity, make_run: RunFactory
) -> None:
    """A newer FINISHED run lacking a checkpoint does not shadow an older complete one."""
    older = make_run(identity, start=1000)
    make_run(identity, start=2000)
    store = MlflowIdentityStore(
        tracking_uri=mlflow_tracking_uri, experiment=EXPERIMENT, require_checkpoint=True
    )
    with patch(
        "neuralls.platform.tracking.mlflow_store.has_checkpoint_artifact",
        side_effect=lambda run_id, **_: run_id == older,
    ):
        assert store.find(identity) == Reused(run_id=older)


def test_walk_is_not_capped_by_page_size(
    store: MlflowIdentityStore,
    mlflow_tracking_uri: str,
    identity: StageIdentity,
    make_run: RunFactory,
) -> None:
    """More broken newer runs than one page cannot hide an older good run."""
    good = make_run(identity, start=1)
    for offset in range(3):
        make_run(identity, start=100 + offset)
    strict = MlflowIdentityStore(
        tracking_uri=mlflow_tracking_uri, experiment=EXPERIMENT, require_checkpoint=True
    )
    with (
        patch("neuralls.platform.tracking.mlflow_store._PAGE_SIZE", 2),
        patch(
            "neuralls.platform.tracking.mlflow_store.has_checkpoint_artifact",
            side_effect=lambda run_id, **_: run_id == good,
        ),
    ):
        assert strict.find(identity) == Reused(run_id=good)


def test_identity_tags_use_the_shared_enum(identity: StageIdentity) -> None:
    """Written tag names are exactly the enum members the store filters on."""
    assert {IdentityTag.STAGE, IdentityTag.KEY, IdentityTag.COMPONENTS} == set(identity.tags())


@pytest.fixture
def reuse_workload(prepared_method: PreparedMethodFactory) -> CompletedMethodWorkload:
    """A one-system converged jacobi workload on dataset ``ds``."""
    method = PreconditionerName("jacobi")
    planned = (SystemTag(DatasetId("ds"), GAUSSIAN, MatrixId(0), 0, SystemId("s0")),)
    setup = MatrixSetupEvent(
        SetupEventId("setup-0"),
        DatasetId("ds"),
        MatrixId(0),
        method,
        1.0,
        (1.0,),
        MeasuredPeakMemory(1),
        SetupCompleted(),
    )
    attempt = SolveAttempt(
        SetupEventId("setup-0"),
        SystemId("s0"),
        DatasetId("ds"),
        GAUSSIAN,
        MatrixId(0),
        0,
        method,
        2.0,
        (2.0,),
        MeasuredPeakMemory(1),
        SolveConverged(2, 1e-8),
    )
    return CompletedMethodWorkload(
        method=method,
        planned=planned,
        setup_events=(setup,),
        solve_attempts=(attempt,),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(2),
        config_digest=CONFIG_DIGEST,
        method_config={"name": "jacobi"},
        presentation=ExecutedMethod(prepared_method(str(method)), ()),
    )


@pytest.fixture
def write_artifacts(
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> Callable[[CompletedMethodWorkload, Path], None]:
    """Write the complete artifact set of a workload into a directory."""

    def write(workload: CompletedMethodWorkload, root: Path) -> None:
        summary = summarize_method_workload(
            workload.method,
            workload.planned,
            workload.setup_events,
            workload.solve_attempts,
            workload.workload_peak_memory,
        )
        write_method_benchmark_artifacts(
            workload,
            summary,
            root,
            systems=provenance_for(workload.planned),
            datasets=group_datasets,
        )

    return write


@pytest.fixture
def method_artifacts(
    tmp_path: Path,
    reuse_workload: CompletedMethodWorkload,
    write_artifacts: Callable[[CompletedMethodWorkload, Path], None],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Create one complete method artifact set for content-validation tests."""
    write_artifacts(reuse_workload, tmp_path)
    paths = {name: str(tmp_path / name) for name in MethodBenchmarkArtifacts.file_names()}
    return paths, ("s0",)


def _log_method_run(
    client: MlflowClient,
    source: Path,
    *,
    tags: dict[str, str],
    names: tuple[str, ...],
) -> str:
    """Create a FINISHED method child holding the named artifacts from ``source``."""
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
    run = client.create_run(experiment_id, tags=tags)
    for name in names:
        client.log_artifact(run.info.run_id, str(source / name), artifact_path=METHOD_ARTIFACT_DIR)
    client.set_terminated(run.info.run_id, status="FINISHED")
    return run.info.run_id


def _find_reusable(
    client: MlflowClient,
    *,
    config_digest: str,
    system_ids: tuple[str, ...],
    method_config: Mapping[str, Json],
) -> Reused | None:
    return find_reusable_benchmark_method(
        tracking_uri=client.tracking_uri,
        experiment=EXPERIMENT,
        method="jacobi",
        config_digest=config_digest,
        system_ids=system_ids,
        method_config=method_config,
    )


def test_reuse_accepts_valid_required_artifact_contents(
    method_artifacts: tuple[dict[str, str], tuple[str, ...]],
) -> None:
    paths, system_ids = method_artifacts

    _validate_method_artifacts(
        paths,
        method="jacobi",
        config_digest=CONFIG_DIGEST,
        system_ids=system_ids,
        method_config={"name": "jacobi"},
    )


def test_reuse_rejects_planned_systems_differing_from_requested_ids(
    method_artifacts: tuple[dict[str, str], tuple[str, ...]],
) -> None:
    paths, _ = method_artifacts

    with pytest.raises(ValueError, match="planned systems do not match requested workload"):
        _validate_method_artifacts(
            paths,
            method="jacobi",
            config_digest=CONFIG_DIGEST,
            system_ids=("s1",),
            method_config={"name": "jacobi"},
        )


def test_reuse_rejects_corrupt_required_presentation(
    method_artifacts: tuple[dict[str, str], tuple[str, ...]],
) -> None:
    paths, system_ids = method_artifacts
    Path(paths["presentation.toml"]).write_text("{", encoding="utf-8")

    with pytest.raises(ValueError):
        _validate_method_artifacts(
            paths,
            method="jacobi",
            config_digest=CONFIG_DIGEST,
            system_ids=system_ids,
            method_config={"name": "jacobi"},
        )


def test_reuse_rejects_presentation_of_another_method(
    method_artifacts: tuple[dict[str, str], tuple[str, ...]],
) -> None:
    paths, system_ids = method_artifacts

    with pytest.raises(ValueError, match="stored presentation does not belong to requested method"):
        _validate_method_artifacts(
            paths,
            method="amg",
            config_digest=CONFIG_DIGEST,
            system_ids=system_ids,
            method_config={"name": "jacobi"},
        )


def test_reuse_misses_a_run_with_planned_systems_differing_from_requested_ids(
    tmp_path: Path,
    client: MlflowClient,
    reuse_workload: CompletedMethodWorkload,
    write_artifacts: Callable[[CompletedMethodWorkload, Path], None],
) -> None:
    """Tags claim ``s1`` while the stored planned systems are ``s0``."""
    write_artifacts(reuse_workload, tmp_path)
    _, tags = build_benchmark_method_run_spec(
        prepared=reuse_workload.presentation.prepared,
        notation=NotationPolicy.default(),
        config_digest=CONFIG_DIGEST,
        system_ids=("s1",),
    )
    _log_method_run(
        client,
        tmp_path,
        tags=tags.as_mlflow_tags(),
        names=MethodBenchmarkArtifacts.file_names(),
    )

    assert (
        _find_reusable(
            client,
            config_digest=CONFIG_DIGEST,
            system_ids=("s1",),
            method_config=reuse_workload.method_config,
        )
        is None
    )


@pytest.mark.parametrize("missing", ("planned_systems.parquet", "presentation.toml"))
def test_reuse_misses_a_run_lacking_a_required_artifact(
    tmp_path: Path,
    client: MlflowClient,
    reuse_workload: CompletedMethodWorkload,
    write_artifacts: Callable[[CompletedMethodWorkload, Path], None],
    missing: str,
) -> None:
    write_artifacts(reuse_workload, tmp_path)
    _, tags = build_benchmark_method_run_spec(
        prepared=reuse_workload.presentation.prepared,
        notation=NotationPolicy.default(),
        config_digest=CONFIG_DIGEST,
        system_ids=("s0",),
    )
    present = tuple(name for name in MethodBenchmarkArtifacts.file_names() if name != missing)
    _log_method_run(client, tmp_path, tags=tags.as_mlflow_tags(), names=present)

    assert (
        _find_reusable(
            client,
            config_digest=CONFIG_DIGEST,
            system_ids=("s0",),
            method_config=reuse_workload.method_config,
        )
        is None
    )


def test_reuse_ignores_checkpoint_lease_path_but_rejects_changed_content(
    tmp_path: Path,
    client: MlflowClient,
    reuse_workload: CompletedMethodWorkload,
    write_artifacts: Callable[[CompletedMethodWorkload, Path], None],
) -> None:
    """Equivalent checkpoint leases reuse one child; changed bytes miss."""
    first_path = tmp_path / "lease-a" / "model.pt"
    second_path = tmp_path / "lease-b" / "model.pt"
    changed_path = tmp_path / "lease-c" / "model.pt"
    for path in (first_path, second_path, changed_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_bytes(b"weights-v1")
    second_path.write_bytes(b"weights-v1")
    changed_path.write_bytes(b"weights-v2")
    first_config = NeuralPreconditionerConfig(checkpoint_path=first_path)
    second_config = NeuralPreconditionerConfig(checkpoint_path=second_path)
    changed_config = NeuralPreconditionerConfig(checkpoint_path=changed_path)
    assert canonical_method_config(first_config) == canonical_method_config(second_config)
    assert canonical_method_config(first_config) != canonical_method_config(changed_config)
    config_digest = canonical_digest(first_config)

    workload = replace(
        reuse_workload,
        config_digest=config_digest,
        method_config=canonical_method_config(first_config),
    )
    method_artifacts = tmp_path / "fixture-artifacts"
    write_artifacts(workload, method_artifacts)
    _, tags = build_benchmark_method_run_spec(
        prepared=workload.presentation.prepared,
        notation=NotationPolicy.default(),
        config_digest=config_digest,
        system_ids=("s0",),
    )
    _log_method_run(
        client,
        method_artifacts,
        tags=tags.as_mlflow_tags(),
        names=MethodBenchmarkArtifacts.file_names(),
    )

    first_hit = _find_reusable(
        client,
        config_digest=config_digest,
        system_ids=("s0",),
        method_config=canonical_method_config(first_config),
    )
    second_hit = _find_reusable(
        client,
        config_digest=canonical_digest(second_config),
        system_ids=("s0",),
        method_config=canonical_method_config(second_config),
    )
    changed_miss = _find_reusable(
        client,
        config_digest=canonical_digest(changed_config),
        system_ids=("s0",),
        method_config=canonical_method_config(changed_config),
    )
    assert first_hit == second_hit
    assert first_hit is not None
    assert changed_miss is None


def test_reuse_rejects_wrong_required_config_contents(
    method_artifacts: tuple[dict[str, str], tuple[str, ...]],
) -> None:
    paths, system_ids = method_artifacts
    Path(paths["method_config.toml"]).write_text('name = "other"\n', encoding="utf-8")

    with pytest.raises(ValueError):
        _validate_method_artifacts(
            paths,
            method="jacobi",
            config_digest="sha256:" + "a" * 64,
            system_ids=system_ids,
            method_config={"name": "jacobi"},
        )


def test_reuse_rejects_incomplete_missing_attempts(
    method_artifacts: tuple[dict[str, str], tuple[str, ...]],
) -> None:
    """A setup failure remains diagnostic-only and never qualifies as reusable."""
    paths, system_ids = method_artifacts
    summary = tomllib.loads(Path(paths["summary.toml"]).read_text(encoding="utf-8"))
    summary["counts"]["missing_system_ids"] = [system_ids[0]]
    summary["counts"]["attempted_rhs"] = len(system_ids) - 1
    Path(paths["summary.toml"]).write_bytes(tomli_w.dumps(summary).encode("utf-8"))

    with pytest.raises(ValueError):
        _validate_method_artifacts(
            paths,
            method="jacobi",
            config_digest="sha256:" + "a" * 64,
            system_ids=system_ids,
            method_config={"name": "jacobi"},
        )


def test_failed_solve_frame_rejects_convergence_only_residual() -> None:
    frame = pl.DataFrame(
        {
            "setup_event_id": ["setup"],
            "system_id": ["system"],
            "dataset": ["ds"],
            "recipe": [recipe_key(GAUSSIAN)],
            "matrix_id": [0],
            "rhs_ordinal": [0],
            "method": ["jacobi"],
            "solve_s": [1.0],
            "samples_s": [[1.0]],
            "peak_memory_bytes": [None],
            "memory_status": ["unknown"],
            "solve_status": ["failed"],
            "iterations": [1],
            "final_relative_residual": [float("nan")],
            "failure_reason": ["limit"],
        }
    )

    with pytest.raises(ValueError):
        validate_decoded_solve_frame(frame)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["memory"].update(status="measured", peak_bytes=-1),
        lambda payload: payload["memory"].pop("status"),
    ],
)
def test_reuse_rejects_invalid_summary_memory(
    method_artifacts: tuple[dict[str, str], tuple[str, ...]],
    mutate: Callable[[dict[str, object]], None],
) -> None:
    paths, system_ids = method_artifacts
    summary = tomllib.loads(Path(paths["summary.toml"]).read_text(encoding="utf-8"))
    mutate(summary)
    Path(paths["summary.toml"]).write_bytes(tomli_w.dumps(summary).encode("utf-8"))

    with pytest.raises((TypeError, ValueError)):
        _validate_method_artifacts(
            paths,
            method="jacobi",
            config_digest="sha256:" + "a" * 64,
            system_ids=system_ids,
            method_config={"name": "jacobi"},
        )
