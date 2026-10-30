"""Checkpoints are loaded once per method workload, before timed builds, and borrowed by builds."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest
import torch
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter

from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
)
from neuralls.composition.comparison.comparison_run import execute_method_workload
from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
    MatrixWorkload,
    ResolvedMethod,
)
from neuralls.composition.preconditioners import preloading
from neuralls.composition.preconditioners.builders import _BuildDeps, _create_predictor
from neuralls.composition.preconditioners.checkpoint_store import (
    CheckpointStore,
    PodBasisKey,
    PredictorKey,
)
from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.composition.preconditioners.preloading import CheckpointLoaders, preload_checkpoints
from neuralls.domain.presentation.provenance import DatasetPresentation
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
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.timing import TimingRepeatPolicy
from neuralls.shared.types import DatasetId
from tests.composition.workload_support import matrix_bundle
from tests.presentation_support import PreparedMethodFactory

pytestmark = pytest.mark.deferred_io("config-loading")

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
def checkpoint() -> Path:
    return Path("model.ckpt")


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


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every preload and build the method workload's service performs, in order."""
    recorded: list[str] = []
    real_create = PreconditionerService.create_preconditioner

    def preload(self: PreconditionerService, matrix: torch.Tensor, config: object) -> None:
        recorded.append("preload")

    def create(
        self: PreconditionerService, matrix: torch.Tensor, config: StandardPreconditionerConfig
    ):
        recorded.append("build")
        return real_create(self, matrix, config)

    monkeypatch.setattr(PreconditionerService, "preload", preload)
    monkeypatch.setattr(PreconditionerService, "create_preconditioner", create)
    return recorded


@pytest.fixture
def slow_preload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every preload take ``SLOW_LOAD_SECONDS``, as a checkpoint read would."""
    monkeypatch.setattr(
        PreconditionerService,
        "preload",
        lambda self, matrix, config: time.sleep(SLOW_LOAD_SECONDS),
    )


@pytest.fixture
def jacobi() -> ResolvedMethod:
    return ResolvedMethod(StandardPreconditionerConfig(type=PreconditionerType.JACOBI))


def _solver_params(repeats: int) -> SolverParams:
    return SolverParams(
        rtol=1e-6,
        atol=1e-10,
        max_iterations=50,
        stopping_criterion="residual_norm",
        m_max=10,
        timing=TimingRepeatPolicy(repeats=repeats),
    )


def _run(
    method: ResolvedMethod,
    bundles: tuple[MatrixWorkload, ...],
    *,
    repeats: int,
    settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    datasets: dict[DatasetId, DatasetPresentation],
) -> CompletedMethodWorkload:
    return execute_method_workload(
        method,
        bundles,
        prepared=prepared_method(str(method.name)),
        datasets=datasets,
        notation=NotationPolicy.default(),
        params=_solver_params(repeats),
        settings=settings,
    )


def test_preload_runs_once_per_matrix_before_all_its_timed_builds(
    events: list[str],
    jacobi: ResolvedMethod,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    _run(
        jacobi,
        (matrix_bundle(1), matrix_bundle(2)),
        repeats=3,
        settings=neuralls_settings,
        prepared_method=prepared_method,
        datasets=group_datasets,
    )

    assert events == ["preload", *["build"] * 3, "preload", *["build"] * 3]


def test_setup_cost_excludes_the_preload(
    slow_preload: None,
    jacobi: ResolvedMethod,
    neuralls_settings: NeurallsSettings,
    prepared_method: PreparedMethodFactory,
    group_datasets: dict[DatasetId, DatasetPresentation],
) -> None:
    workload = _run(
        jacobi,
        (matrix_bundle(1),),
        repeats=1,
        settings=neuralls_settings,
        prepared_method=prepared_method,
        datasets=group_datasets,
    )

    assert workload.setup_events[0].setup_s < SLOW_LOAD_SECONDS
