"""Checkpoints are loaded once per method workload, before timed builds, and borrowed by builds."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest
import torch
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter

from neuralls.composition.comparison._preconditioner_setup import (
    DatasetPreconditionerCache,
    DatasetScope,
    PreconditionerService,
)
from neuralls.composition.comparison.comparison_run import _run_preconditioner
from neuralls.composition.preconditioners import preloading
from neuralls.composition.preconditioners.builders import _BuildDeps, _create_predictor
from neuralls.composition.preconditioners.checkpoint_store import (
    CheckpointStore,
    PodBasisKey,
    PredictorKey,
)
from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.composition.preconditioners.preloading import CheckpointLoaders, preload_checkpoints
from neuralls.domain.solver.models.config import SolverParams
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralCheckpointRef,
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PODCoarseningConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.dlkit.predictor_adapter import OwningDLKitNeuralPreconditioner
from neuralls.shared.timing import TimingRepeatPolicy
from neuralls.shared.types import MatrixId

POD_RANK = 2
SLOW_LOAD_SECONDS = 0.2


class FakePredictor(ExtraInputPredictorPort):
    """Records how often it is released."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.cleanups = 0

    @property
    def required_inputs(self) -> tuple[str, ...]:
        return ()

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        return residual

    def cleanup(self) -> None:
        self.cleanups += 1


class RecordingAdapter(PredictorAdapter):
    """Adapter double that remembers every checkpoint it was asked to load."""

    def __init__(self) -> None:
        self.created: list[FakePredictor] = []

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> FakePredictor:
        predictor = FakePredictor(checkpoint_path)
        self.created.append(predictor)
        return predictor


@pytest.fixture
def adapter() -> RecordingAdapter:
    return RecordingAdapter()


@pytest.fixture
def loaders(adapter: RecordingAdapter) -> CheckpointLoaders:
    def no_inference_predictor(path: Path) -> None:
        raise AssertionError(f"unexpected inference predictor for {path}")

    return CheckpointLoaders(adapter, no_inference_predictor)  # type: ignore[arg-type]


@pytest.fixture
def matrix() -> torch.Tensor:
    return torch.eye(3, dtype=torch.float64)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"placeholder")
    return path


def test_store_loads_each_key_once_and_releases_each_predictor_once(checkpoint: Path) -> None:
    store = CheckpointStore()
    adapter = RecordingAdapter()
    key = PredictorKey(checkpoint)

    first = store.load_predictor(key, lambda: adapter.create_predictor(checkpoint))
    second = store.load_predictor(key, lambda: adapter.create_predictor(checkpoint))
    store.release()

    assert first is second
    assert len(adapter.created) == 1
    assert adapter.created[0].cleanups == 1
    assert store.predictor(key) is None


def test_a_built_preconditioner_gc_does_not_unload_the_shared_predictor(
    checkpoint: Path, matrix: torch.Tensor
) -> None:
    """A discarded timing repeat must not unload the predictor the kept build still uses."""
    store = CheckpointStore()
    adapter = RecordingAdapter()
    cfg = NeuralPreconditionerConfig(checkpoint_path=checkpoint)
    preload_checkpoints(cfg, matrix, store, CheckpointLoaders(adapter, lambda path: None))  # type: ignore[arg-type,return-value]

    discarded = create_preconditioner(matrix, cfg, preloaded=store)
    kept = create_preconditioner(matrix, cfg, preloaded=store)
    del discarded
    gc.collect()

    assert len(adapter.created) == 1
    assert adapter.created[0].cleanups == 0
    del kept
    gc.collect()
    store.release()
    assert adapter.created[0].cleanups == 1


def test_neural_amg_transfer_predictors_are_borrowed_from_the_store(
    tmp_path: Path, matrix: torch.Tensor
) -> None:
    """Prolongation and restriction each get their own preloaded predictor, never a load."""
    prolongation = NeuralTransferConfig(checkpoint_path=tmp_path / "p.ckpt")
    restriction = NeuralTransferConfig(checkpoint_path=tmp_path / "r.ckpt")
    cfg = NeuralAMGPreconditionerConfig(prolongation=prolongation, restriction=restriction)
    store = CheckpointStore()
    loading = RecordingAdapter()
    preload_checkpoints(cfg, matrix, store, CheckpointLoaders(loading, lambda path: None))  # type: ignore[arg-type,return-value]
    failing_adapter = RecordingAdapter()
    deps = _BuildDeps(adapter=failing_adapter, inference_predictor_factory=None, preloaded=store)

    borrowed_p = _create_predictor(prolongation, tmp_path / "p.ckpt", deps)
    borrowed_r = _create_predictor(restriction, tmp_path / "r.ckpt", deps)

    assert isinstance(borrowed_p, FakePredictor)
    assert isinstance(borrowed_r, FakePredictor)
    assert (borrowed_p.path.name, borrowed_r.path.name) == ("p.ckpt", "r.ckpt")
    assert failing_adapter.created == []


def test_without_a_preload_the_built_preconditioner_owns_and_releases_its_predictor(
    checkpoint: Path, matrix: torch.Tensor
) -> None:
    adapter = RecordingAdapter()
    cfg = NeuralPreconditionerConfig(checkpoint_path=checkpoint)

    precond = create_preconditioner(matrix, cfg, adapter=adapter)
    assert isinstance(precond, OwningDLKitNeuralPreconditioner)
    precond.cleanup()
    precond.cleanup()

    assert len(adapter.created) == 1
    assert adapter.created[0].cleanups == 2  # the predictor's own cleanup is idempotent


def test_preload_covers_every_checkpoint_of_a_neural_amg(
    tmp_path: Path, matrix: torch.Tensor, adapter: RecordingAdapter, loaders: CheckpointLoaders
) -> None:
    cfg = NeuralAMGPreconditionerConfig(
        prolongation=NeuralTransferConfig(checkpoint_path=tmp_path / "p.ckpt"),
        restriction=NeuralTransferConfig(checkpoint_path=tmp_path / "r.ckpt"),
    )
    store = CheckpointStore()

    preload_checkpoints(cfg, matrix, store, loaders)
    preload_checkpoints(cfg, matrix, store, loaders)

    assert sorted(p.path.name for p in adapter.created) == ["p.ckpt", "r.ckpt"]


def test_preload_is_a_no_op_for_configs_without_checkpoints(
    matrix: torch.Tensor, adapter: RecordingAdapter, loaders: CheckpointLoaders
) -> None:
    store = CheckpointStore()

    preload_checkpoints(
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI), matrix, store, loaders
    )

    assert adapter.created == []


def test_a_fitted_pod_basis_is_read_once_and_shared_by_every_matrix(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, loaders: CheckpointLoaders
) -> None:
    reads: list[Path] = []

    def fake_read(path: Path) -> torch.nn.Module:
        reads.append(path)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(preloading, "load_fitted_pod_model", fake_read)
    cfg = AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(
            checkpoint_path=checkpoint, rank=POD_RANK, dataset_dir=checkpoint.parent
        ),
    )
    store = CheckpointStore()
    matrices = [torch.eye(n, dtype=torch.float64) for n in (3, 4, 5)]

    for matrix in matrices:
        preload_checkpoints(cfg, matrix, store, loaders)

    key = PodBasisKey(checkpoint, torch.float64, torch.device("cpu"))
    assert reads == [checkpoint]
    basis = store.pod_basis(key)
    assert isinstance(basis, torch.nn.Linear)
    assert basis.weight.dtype is torch.float64


def test_an_unmapped_checkpoint_kind_fails_loudly(
    matrix: torch.Tensor, loaders: CheckpointLoaders, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preloading, "_REF_PRELOADERS", {})
    cfg = NeuralPreconditionerConfig(checkpoint_path=Path("model.ckpt"))

    with pytest.raises(TypeError):
        preload_checkpoints(cfg, matrix, CheckpointStore(), loaders)


def test_every_checkpoint_ref_kind_has_a_preloader() -> None:
    def concrete_subclasses(cls: type) -> set[type]:
        found: set[type] = set()
        for sub in cls.__subclasses__():
            found.add(sub)
            found |= concrete_subclasses(sub)
        return found

    ref_kinds = concrete_subclasses(NeuralCheckpointRef)

    assert ref_kinds <= set(preloading._REF_PRELOADERS)


class OrderRecordingService:
    """Service double that records preload/build order and lets preload be slow."""

    def __init__(self, preload_seconds: float = 0.0) -> None:
        self.events: list[str] = []
        self._preload_seconds = preload_seconds
        self._inner = PreconditionerService()

    def preload(self, matrix: torch.Tensor, config: StandardPreconditionerConfig) -> None:
        self.events.append("preload")
        time.sleep(self._preload_seconds)

    def create_preconditioner(self, matrix: torch.Tensor, config: StandardPreconditionerConfig):
        self.events.append("build")
        return self._inner.create_preconditioner(matrix, config)


def _run(service: OrderRecordingService, settings: NeurallsSettings, repeats: int) -> float:
    entry = _run_preconditioner(
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI),
        service=service,  # type: ignore[arg-type]
        matrix=torch.diag(torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)),
        rhs=torch.ones(3, dtype=torch.float64),
        matrix_path=Path("/nonexistent"),
        matrix_index=0,
        params=SolverParams(
            rtol=1e-6,
            atol=1e-10,
            max_iterations=50,
            stopping_criterion="residual_norm",
            m_max=10,
            timing=TimingRepeatPolicy(repeats=repeats),
        ),
        color_key=None,
        marker_key=None,
        x_exact=None,
        settings=settings,
    )
    return entry.result.setup_cost


def test_preload_runs_once_before_all_timed_builds(neuralls_settings: NeurallsSettings) -> None:
    service = OrderRecordingService()

    _run(service, neuralls_settings, repeats=3)

    assert service.events == ["preload", "build", "build", "build"]


def test_setup_cost_excludes_the_preload(neuralls_settings: NeurallsSettings) -> None:
    service = OrderRecordingService(preload_seconds=SLOW_LOAD_SECONDS)

    setup_cost = _run(service, neuralls_settings, repeats=1)

    assert setup_cost < SLOW_LOAD_SECONDS


def test_matrix_dependent_builds_are_evicted_but_static_ones_and_checkpoints_stay() -> None:
    scope = DatasetScope()
    first, second = MatrixId(0), MatrixId(1)
    jacobi = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)
    matrix = torch.eye(3, dtype=torch.float64)
    for matrix_id in (first, second):
        view = scope.for_system(matrix_id)
        service = PreconditionerService(checkpoints=scope.checkpoints)
        view.put(
            "jacobi",
            service.create_preconditioner(matrix, jacobi),
            setup_cost_seconds=0.0,
            setup_peak_memory_bytes=None,
        )
    view = scope.for_system(first)
    scope.checkpoints.load_predictor(
        PredictorKey(Path("m.ckpt")), lambda: FakePredictor(Path("m.ckpt"))
    )

    scope.cache.evict_matrix(first)

    assert view.get("jacobi") is None
    assert scope.for_system(second).get("jacobi") is not None
    assert scope.checkpoints.predictor(PredictorKey(Path("m.ckpt"))) is not None
    scope.release()
    assert scope.checkpoints.predictor(PredictorKey(Path("m.ckpt"))) is None


def test_dataset_cache_stays_usable_without_a_checkpoint_store() -> None:
    cache = DatasetPreconditionerCache().for_system(MatrixId(0))

    assert cache.checkpoints is None
