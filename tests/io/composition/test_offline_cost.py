"""Tests for `_offline_cost.py`: the generation cost, the training cost and their pairing.

A comparison run never generates a dataset itself -- this module reads back
the dataset's own manifest (`DatasetManifest.generation_duration_seconds`,
stamped once at generation time) to answer "how much did generating this
preconditioner's training data cost, and do we still know that?" These tests
cover every dispatch branch in `_resolve_generation_dataset_dirs`: classical
AMG (not applicable), POD-2G (`dataset_dir` directly), and a checkpoint-backed
Neural config (`data_config_path` resolved the same way
`training_batch.py::run_assignment` already does) -- the gap the design
investigation found (a plain Neural preconditioner's generation cost was
previously unresolvable at all).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from neuralls.composition.comparison._offline_cost import (
    resolve_generation_cost,
    resolve_offline_cost,
    resolve_training_cost,
)
from neuralls.domain.solver.models.cost import Known, NotApplicable, OfflineStages, Unavailable
from neuralls.platform.config.models.coarsening import (
    AggregationCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PreconditionerType,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest as _make_manifest
from neuralls.platform.storage.manifest_io import save_dataset_manifest
from neuralls.platform.tracking.services import MlflowTrackingServices
from neuralls.shared.types import CostProvenance
from tests.fakes.mlflow_client import FakeMlflowClient

_GENERATION_DURATION = 12.5
WORKSPACE = Path("/workspace")


def _write_data_config(path: Path, data_root: Path, *, dataset_id: str = "test-dataset") -> None:
    """Write a minimal data config TOML resolving to `data_root / dataset_id`."""
    path.write_text(
        "\n".join(
            [
                f'id = "{dataset_id}"',
                "",
                "[output]",
                f'data_dir = "{data_root.as_posix()}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _stamp_manifest(dataset_dir: Path, *, generation_duration_seconds: float | None) -> None:
    """Write a minimal, real dataset manifest at `dataset_dir`."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    artifact = DatasetArtifact(path="matrix.npy", format="npy", dtype="float64", shape=(10, 10))
    vector_artifact = DatasetArtifact(path="rhs.npy", format="npy", dtype="float64", shape=(10,))
    manifest = _make_manifest(
        matrix=artifact,
        rhs=vector_artifact,
        solutions=vector_artifact,
        normalization=DatasetNormalization(
            type="none", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
        ),
        generation_duration_seconds=generation_duration_seconds,
        generation_peak_memory_bytes=(1024 if generation_duration_seconds is not None else None),
    )
    save_dataset_manifest(dataset_dir, manifest)


def test_classical_amg_has_no_generation_cost(neuralls_settings: NeurallsSettings) -> None:
    """Classical/geometric AMG has no dataset field at all -- not applicable."""
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG, coarsening=AggregationCoarseningConfig()
    )
    assert resolve_generation_cost(cfg, settings=neuralls_settings) == NotApplicable()


def test_pod2g_with_stamped_manifest_reports_historical_cost(
    tmp_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """A POD-2G config's `dataset_dir` manifest with a real stamped duration
    resolves to a HISTORICAL StageCost with that exact wall time.
    """
    dataset_dir = tmp_path / "pod-dataset"
    _stamp_manifest(dataset_dir, generation_duration_seconds=_GENERATION_DURATION)
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=4),
    )

    result = resolve_generation_cost(cfg, settings=neuralls_settings)

    assert result == Known(
        wall_time_seconds=_GENERATION_DURATION,
        peak_memory_bytes=1024,
        provenance=CostProvenance.HISTORICAL,
    )


def test_pod2g_manifest_missing_duration_field_is_unavailable(
    tmp_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """A manifest that exists but predates `generation_duration_seconds` is
    UNAVAILABLE, not a crash and not a silent zero.
    """
    dataset_dir = tmp_path / "pod-dataset"
    _stamp_manifest(dataset_dir, generation_duration_seconds=None)
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=4),
    )

    assert isinstance(resolve_generation_cost(cfg, settings=neuralls_settings), Unavailable)


def test_pod2g_missing_dataset_dir_is_unavailable_not_an_exception(
    tmp_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """A `dataset_dir` pointing at a directory with no manifest at all (never
    generated) must resolve to UNAVAILABLE, never raise.
    """
    dataset_dir = tmp_path / "never-generated"
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=4),
    )

    assert isinstance(resolve_generation_cost(cfg, settings=neuralls_settings), Unavailable)


def test_neural_config_with_data_config_path_reports_historical_cost(
    tmp_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """Closes the gap: a checkpoint-bearing Neural config's `data_config_path`
    resolves (via `load_data_config`/`resolve_dataset_identity`, exactly like
    `training_batch.py::run_assignment`) to a real dataset dir, whose stamped
    manifest reports a HISTORICAL generation cost -- previously unresolvable
    for plain Neural preconditioners.
    """
    data_root = tmp_path / "data-root"
    dataset_dir = data_root / "test-dataset"
    _stamp_manifest(dataset_dir, generation_duration_seconds=_GENERATION_DURATION)

    data_config_path = tmp_path / "data_config.toml"
    _write_data_config(data_config_path, data_root)

    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        data_config_path=data_config_path,
    )

    result = resolve_generation_cost(cfg, settings=neuralls_settings)
    assert isinstance(result, Known)

    assert result.provenance is CostProvenance.HISTORICAL
    assert result.wall_time_seconds == _GENERATION_DURATION


def test_neural_config_without_data_config_path_has_no_generation_cost(
    neuralls_settings: NeurallsSettings,
) -> None:
    """A Neural config with no `data_config_path` at all is not applicable."""
    cfg = NeuralPreconditionerConfig(type=PreconditionerType.NEURAL)
    assert resolve_generation_cost(cfg, settings=neuralls_settings) == NotApplicable()


TRAINING_START_MS = 10_000
TRAINING_SECONDS = 90.5
OTHER_TRAINING_SECONDS = 30.0


@pytest.fixture
def fake_store(tmp_path: Path) -> FakeMlflowClient:
    """In-memory MLflow store holding the training runs."""
    return FakeMlflowClient(artifact_root=tmp_path / "artifacts")


@pytest.fixture
def runs(fake_store: FakeMlflowClient) -> MlflowClient:
    """A client over the fake store, as composition obtains one from its tracking services."""
    return MlflowTrackingServices(client_factory=fake_store).client(fake_store.tracking_uri)


def _finished_run(store: FakeMlflowClient, name: str, seconds: float) -> str:
    return store.add_run(
        name,
        start_time_ms=TRAINING_START_MS,
        end_time_ms=TRAINING_START_MS + round(seconds * 1000),
    )


def _trained(run_id: str | None) -> NeuralPreconditionerConfig:
    return NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        checkpoint_path=WORKSPACE / "model.ckpt",
        resolved_run_id=run_id,
    )


def test_classical_amg_has_no_training_cost(runs: MlflowClient) -> None:
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG, coarsening=AggregationCoarseningConfig()
    )

    assert resolve_training_cost(cfg, runs=runs) == NotApplicable()


def test_pod2g_fit_inline_has_no_training_stage(runs: MlflowClient) -> None:
    """The inline basis fit runs inside setup, which is already a measured online cost."""
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(dataset_dir=WORKSPACE / "pod-dataset", rank=4),
    )

    assert resolve_training_cost(cfg, runs=runs) == NotApplicable()


def test_checkpoint_training_cost_is_the_wall_time_of_its_run(
    fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    run_id = _finished_run(fake_store, "trained", TRAINING_SECONDS)

    assert resolve_training_cost(_trained(run_id), runs=runs) == Known(
        TRAINING_SECONDS, None, CostProvenance.HISTORICAL
    )


def test_pod2g_checkpoint_training_cost_is_the_wall_time_of_its_run(
    fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    run_id = _finished_run(fake_store, "fitted", TRAINING_SECONDS)
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(
            dataset_dir=WORKSPACE / "pod-dataset",
            rank=4,
            checkpoint_path=WORKSPACE / "pod.ckpt",
            resolved_run_id=run_id,
        ),
    )

    assert resolve_training_cost(cfg, runs=runs) == Known(
        TRAINING_SECONDS, None, CostProvenance.HISTORICAL
    )


def test_prolongation_and_restriction_trained_apart_add_their_runs(
    fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    prolongation_run = _finished_run(fake_store, "prolongation", TRAINING_SECONDS)
    restriction_run = _finished_run(fake_store, "restriction", OTHER_TRAINING_SECONDS)
    cfg = NeuralAMGPreconditionerConfig(
        prolongation=NeuralTransferConfig(
            checkpoint_path=WORKSPACE / "p.ckpt", resolved_run_id=prolongation_run
        ),
        restriction=NeuralTransferConfig(
            checkpoint_path=WORKSPACE / "r.ckpt", resolved_run_id=restriction_run
        ),
    )

    assert resolve_training_cost(cfg, runs=runs) == Known(
        TRAINING_SECONDS + OTHER_TRAINING_SECONDS, None, CostProvenance.HISTORICAL
    )


def test_transfer_operators_trained_in_one_run_cost_that_run_once(
    fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    run_id = _finished_run(fake_store, "both", TRAINING_SECONDS)
    cfg = NeuralAMGPreconditionerConfig(
        prolongation=NeuralTransferConfig(
            checkpoint_path=WORKSPACE / "p.ckpt", resolved_run_id=run_id
        ),
        restriction=NeuralTransferConfig(
            checkpoint_path=WORKSPACE / "r.ckpt", resolved_run_id=run_id
        ),
    )

    assert resolve_training_cost(cfg, runs=runs) == Known(
        TRAINING_SECONDS, None, CostProvenance.HISTORICAL
    )


def test_an_explicit_checkpoint_path_has_no_run_to_read(runs: MlflowClient) -> None:
    assert resolve_training_cost(_trained(None), runs=runs) == Unavailable(
        "1 of 1 checkpoints have no recorded training run"
    )


def test_one_unrecorded_transfer_operator_makes_the_whole_cost_unavailable(
    fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    run_id = _finished_run(fake_store, "prolongation", TRAINING_SECONDS)
    cfg = NeuralAMGPreconditionerConfig(
        prolongation=NeuralTransferConfig(
            checkpoint_path=WORKSPACE / "p.ckpt", resolved_run_id=run_id
        ),
        restriction=NeuralTransferConfig(checkpoint_path=WORKSPACE / "r.ckpt"),
    )

    assert resolve_training_cost(cfg, runs=runs) == Unavailable(
        "1 of 2 checkpoints have no recorded training run"
    )


def test_a_training_run_missing_from_the_store_is_unavailable(runs: MlflowClient) -> None:
    assert resolve_training_cost(_trained("gone"), runs=runs) == Unavailable(
        "run gone cannot be read: Run 'gone' not found"
    )


def test_a_training_run_still_running_is_unavailable(
    fake_store: FakeMlflowClient, runs: MlflowClient
) -> None:
    run_id = fake_store.add_run("running", start_time_ms=TRAINING_START_MS)

    assert resolve_training_cost(_trained(run_id), runs=runs) == Unavailable(
        f"run {run_id} has no end time"
    )


def test_offline_cost_pairs_the_generation_and_training_stages(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
    fake_store: FakeMlflowClient,
    runs: MlflowClient,
) -> None:
    data_root = tmp_path / "data-root"
    _stamp_manifest(data_root / "test-dataset", generation_duration_seconds=_GENERATION_DURATION)
    data_config_path = tmp_path / "data_config.toml"
    _write_data_config(data_config_path, data_root)
    run_id = _finished_run(fake_store, "trained", TRAINING_SECONDS)
    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        data_config_path=data_config_path,
        checkpoint_path=WORKSPACE / "model.ckpt",
        resolved_run_id=run_id,
    )

    assert resolve_offline_cost(cfg, settings=neuralls_settings, runs=runs) == OfflineStages(
        generation=Known(_GENERATION_DURATION, 1024, CostProvenance.HISTORICAL),
        training=Known(TRAINING_SECONDS, None, CostProvenance.HISTORICAL),
    )
