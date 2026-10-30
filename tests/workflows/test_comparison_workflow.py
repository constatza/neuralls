"""Tests for neuralls.composition.assignments.comparison_batch."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tomli_w

from neuralls.composition.assignments.comparison_batch import (
    PendingComparison,
    PreparedComparisonExecution,
    _ComparisonResolutionContext,
    _prepare_comparison_entry,
    _project_setup_measurements,
    _resolve_comparison_topology,
    _resolve_neural_preconditioners,
    _validate_neural_preconditioner,
    neural_specs_from_assignments,
    run_comparison_batch,
)
from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison.comparison_run import execute_method_workload
from neuralls.composition.comparison.config_assembler import resolve_comparison_config
from neuralls.composition.comparison.models import (
    ComparisonCompleted,
    ComparisonEntryFailed,
    ComparisonOutcome,
    ComparisonParams,
    ComparisonReused,
    CompletedMethodWorkload,
    MatrixRef,
    MatrixWorkload,
    PlannedSystem,
    ResolvedComparisonInput,
    ResolvedMethod,
)
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import ExecutedMethod, PreparedMethodPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import NamedRun
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    MeasuredPeakMemory,
    SetupCompleted,
    SystemTag,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.plot_paths import PlotPaths
from neuralls.domain.solver.models.provenance import (
    HeldOutMatrix,
    MatrixProvenance,
    SharedSingleMatrix,
)
from neuralls.domain.solver.models.recommendations import (
    ComparisonRecommendations,
    RankedRecommendation,
)
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    ComparisonRegistryEntry,
)
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    LoggedModelRefConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
    StandardPreconditionerConfig,
    TrainedAssignmentRefConfig,
)
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.tracking.mlflow import sanitize_metric_key_segment
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import (
    ComparisonRhsSourceKind,
    DatasetId,
    MatrixId,
    RowKind,
    SetupEventId,
    SystemId,
)
from tests.presentation_support import NO_DATASET_READS, PreparedMethodFactory
from tests.recipe_support import GAUSSIAN
from tests.workflows.comparison_support import (
    make_settings,
    mock_cfg,
    write_dataset_config,
    write_experiments_config,
)

pytestmark = pytest.mark.deferred_io("generation-io")


def test_resolve_comparison_config_importable_from_composition() -> None:
    """resolve_comparison_config must live in composition, not platform."""
    from neuralls.composition.comparison.config_assembler import (
        resolve_comparison_config,  # noqa: F401
    )


_RESOLVE_COMPARISON_CONFIG = (
    "neuralls.composition.comparison.config_assembler.resolve_comparison_config"
)
_RESOLVE_MODELS = (
    "neuralls.composition.assignments.comparison_batch.resolve_preconditioner_models_with_warnings"
)
NO_RUNNABLE_PRECONDITIONERS = "No runnable preconditioners remain after model resolution."


def _write_comparison_config(path: Path) -> None:
    matrix_path = path.parent / "matrix.npy"
    rhs_path = path.parent / "rhs.npy"
    path.write_text(
        "\n".join(
            [
                "[general]",
                "",
                "[general.params]",
                "rtol = 1.0e-6",
                "atol = 1.0e-14",
                "max_iterations = 10",
                'stopping_criterion = "residual_norm"',
                "m_max = 20",
                "",
                "[general.data]",
                f'matrix_path = "{matrix_path.as_posix()}"',
                f'rhs_path = "{rhs_path.as_posix()}"',
                "",
                "[[preconditioners]]",
                'name = "identity"',
                'type = "identity"',
            ]
        ),
        encoding="utf-8",
    )


def _write_train_job_config(path: Path) -> None:
    """Write a minimal, fully-loadable `run.type = "train"` job config."""
    path.write_text(
        '[run]\ntype = "train"\n\n[model]\nname = "ScaleEquivariantEmbeddedFactorizedFFNN"\nnum_layers = 1\nactivation = "gelu"\n\n[data]\nname = "FlexibleDataset"\nbatch_size = 4\n\n[data.module]\nname = "ArrayDataModule"\nmodule_path = "dlkit.engine.adapters.lightning.datamodules"\n\n[training]',
        encoding="utf-8",
    )


def _write_fit_job_config(path: Path, *, rank: float = 0.99) -> None:
    """Write a minimal, fully-loadable `run.type = "fit"` (POD-2G) job config."""
    path.write_text(
        "\n".join(
            [
                "[run]",
                'type = "fit"',
                "",
                "[model]",
                'name = "PODCoarseningStrategy"',
                'module_path = "torchalg.preconditioners.implementations.pod"',
                f"rank = {rank}",
                "",
                "[data]",
                'name = "FlexibleDataset"',
                "batch_size = 4",
                "",
                "[data.module]",
                'name = "ArrayDataModule"',
                'module_path = "dlkit.engine.adapters.lightning.datamodules"',
            ]
        ),
        encoding="utf-8",
    )


def _case_config_with_assignment(
    tmp_path: Path,
    *,
    job_writer: Callable[[Path], None],
    assignment_id: str = "ffnn_solutions",
    dataset_id: str = "solutions",
    job_id: str = "ffnn",
) -> tuple[CaseConfig, Path, NeurallsSettings]:
    """Build a minimal case config with one dataset/job/assignment registered.

    Returns the ``(cfg, config_dir, settings)`` triple `neural_specs_from_assignments`
    needs to resolve an assignment's declared job and dataset directory.
    """
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    write_dataset_config(datasets_dir / f"{dataset_id}.toml", dataset_id)

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    job_writer(jobs_dir / f"{job_id}.toml")

    settings = make_settings(tmp_path)
    cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": dataset_id, "path": f"datasets/{dataset_id}.toml"}],
            "jobs": [{"id": job_id, "path": f"jobs/{job_id}.toml"}],
            "assignments": [{"id": assignment_id, "dataset": dataset_id, "job": job_id}],
        }
    )
    return cfg, tmp_path, settings


def _solver_params(tmp_path: Path) -> ComparisonGeneral:
    return ComparisonGeneral(
        params=SolverParams(
            rtol=1.0e-6,
            atol=1.0e-14,
            max_iterations=10,
            stopping_criterion="residual_norm",
            m_max=20,
        ),
        data=ComparisonData(
            matrix_path=tmp_path / "matrix.npy",
            rhs_path=tmp_path / "rhs.npy",
        ),
    )


def _write_system_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal matrix/RHS arrays used by comparison preflight."""
    matrix_path = tmp_path / "matrix.npy"
    rhs_path = tmp_path / "rhs.npy"
    np.save(matrix_path, np.eye(2, dtype=np.float64))
    np.save(rhs_path, np.array([[1.0, 0.0]], dtype=np.float64))
    return matrix_path, rhs_path


def _logged_ref(run_id: str) -> LoggedModelRefConfig:
    return LoggedModelRefConfig(run_id=run_id)


def _registered_ref(name: str, alias: str) -> RegisteredModelRefConfig:
    return RegisteredModelRefConfig(name=name, alias=alias)


def _typed_comparison_result(plot_path: Path) -> ComparisonResult:
    return ComparisonResult(
        results={
            "identity": CGComparisonResult(
                x=np.array([1.0, 2.0]),
                converged=True,
                iterations=2,
                residual=1.0e-8,
                residual_abs=1.0e-9,
                residual_history_rel=[1.0, 1.0e-8],
                residual_history_abs=[1.0, 1.0e-9],
                preconditioner="identity",
                initial_guess=np.zeros(2),
                exact_error=None,
                rhs_norm=1.0,
                breakdown=False,
            )
        },
        summary="ok",
        solver_params=_solver_params(plot_path.parent),
        plot_paths=PlotPaths(convergence=plot_path),
        preconditioners=("identity",),
        recommendations=ComparisonRecommendations(
            ranked=(
                RankedRecommendation(
                    method_id="identity",
                    iterations=2,
                    residual=1.0e-8,
                    residual_abs=1.0e-9,
                    breakdown=False,
                ),
            ),
            overall_best=RankedRecommendation(
                method_id="identity",
                iterations=2,
                residual=1.0e-8,
                residual_abs=1.0e-9,
                breakdown=False,
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _scope(stub_matrix_scope: object) -> None:
    """These tests mock the case registry, so the matrix dataset's presentation is supplied."""


@pytest.fixture(autouse=True)
def stub_comparison_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests drive the workflow with mocked configs; identity derivation has its own tests."""
    monkeypatch.setattr(
        "neuralls.composition.assignments.comparison_batch.comparison_identity",
        lambda *_args, **_kwargs: StageIdentity.build(
            "comparison", {"stub": canonical_digest("stub")}
        ),
    )


def _make_entry(comparison_id: str = "test") -> ComparisonRegistryEntry:
    """Build a minimal ComparisonRegistryEntry for direct preparation calls."""
    return ComparisonRegistryEntry(
        id=comparison_id,
        matrix_dataset="solutions",
        rhs_source={"kind": "gaussian"},
    )


def _planned(
    entry: ComparisonRegistryEntry, *, matrix_row: int = 0, rhs_row: int | None = None
) -> PlannedSystem:
    """Bind an entry to the dataset's one shared matrix, as the planner does for [[comparisons]]."""
    return PlannedSystem(
        entry=entry,
        matrix=MatrixRef(matrix_id=MatrixId(0), row=matrix_row),
        rhs_row=rhs_row,
        rhs_ordinal=0,
        provenance=SharedSingleMatrix(raw_matrix_index=matrix_row),
    )


def test_resolve_comparison_config_binds_the_planned_matrix_row_and_dataset_row(
    tmp_path: Path,
) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    settings = make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_baseline": {"preconditioners": [{"name": "identity", "type": "identity"}]},
        }
    )
    entry = ComparisonRegistryEntry(
        id="dataset-row", matrix_dataset="matrix", rhs_source={"kind": "dataset"}
    )

    cfg = resolve_comparison_config(
        case_cfg, tmp_path, _planned(entry, matrix_row=4, rhs_row=4), settings
    )

    assert cfg.general.data.matrix_path == settings.processed_dir / "matrix"
    assert cfg.general.data.matrix_index == 4
    assert cfg.general.data.rhs_source_params == {"sample_index": 4, "mask": {"kind": "none"}}


def test_resolve_comparison_config_presentation_falls_back_to_case_default(
    tmp_path: Path,
) -> None:
    """With no per-entry method override, the case's `[comparison_baseline.presentation]` applies."""
    from neuralls.shared.scientific_description.policy import RenderTarget

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    settings = make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_baseline": {
                "preconditioners": [{"type": "identity"}],
                "presentation": {"rendering": "latex"},
            },
        }
    )
    entry = ComparisonRegistryEntry(
        id="no-override",
        matrix_dataset="matrix",
        rhs_source={"kind": "dataset"},
    )

    resolved = resolve_comparison_config(case_cfg, tmp_path, _planned(entry), settings)

    assert resolved.notation_policy.target is RenderTarget.LATEX


def test_resolve_comparison_config_presentation_override_replaces_case_default(
    tmp_path: Path,
) -> None:
    """A per-entry `method` file's `[presentation]` fully replaces the case-level default.

    Matches how `preconditioners`/`normalize_system` already override in
    this same function — a full replace, not a field-level merge.
    """
    from neuralls.shared.scientific_description.policy import RenderTarget

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    settings = make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_baseline": {
                "preconditioners": [{"type": "identity"}],
                "presentation": {"rendering": "latex"},
            },
        }
    )
    method_path = tmp_path / "method.toml"
    method_path.write_text(
        tomli_w.dumps({"presentation": {"rendering": "mathtext"}}), encoding="utf-8"
    )
    entry = ComparisonRegistryEntry(
        id="with-override",
        matrix_dataset="matrix",
        method=method_path,
        rhs_source={"kind": "dataset"},
    )

    resolved = resolve_comparison_config(case_cfg, tmp_path, _planned(entry), settings)

    assert resolved.notation_policy.target is RenderTarget.MATHTEXT


def test_reference_validation_waits_for_assignment_methods(
    tmp_path: Path,
) -> None:
    """A reference supplied by later assignment expansion is not rejected early."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    method_path = tmp_path / "method.toml"
    method_path.write_text(
        tomli_w.dumps(
            {
                "general": {"reference_preconditioner": "assignment-model"},
                "preconditioners": [{"type": "identity"}],
            }
        ),
        encoding="utf-8",
    )
    settings = make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_baseline": {"preconditioners": [{"type": "identity"}]},
        }
    )
    entry = ComparisonRegistryEntry(
        id="with-assignment-reference",
        matrix_dataset="matrix",
        method=method_path,
        rhs_source={"kind": "gaussian"},
    )

    resolved = resolve_comparison_config(case_cfg, tmp_path, _planned(entry), settings)

    assert resolved.general.reference_preconditioner == "assignment-model"


def test_resolve_comparison_config_carries_generated_rhs_without_rhs_path(
    tmp_path: Path,
) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    settings = make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_baseline": {
                "rtol": 1.0e-6,
                "atol": 1.0e-14,
                "max_iterations": 10,
                "stopping_criterion": "residual_norm",
                "m_max": 20,
                "preconditioners": [{"name": "identity", "type": "identity"}],
            },
        }
    )
    entry = ComparisonRegistryEntry(
        id="generated",
        matrix_dataset="matrix",
        rhs_source={"kind": "gaussian"},
        seed=42,
    )

    cfg = resolve_comparison_config(case_cfg, tmp_path, _planned(entry), settings)

    assert cfg.general.data.rhs_path is None
    assert cfg.general.data.rhs_source_kind is ComparisonRhsSourceKind.GAUSSIAN
    assert cfg.general.data.selection_seed == 42


def test_resolve_comparison_config_carries_raw_rhs_source(
    tmp_path: Path,
) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    rhs_path = tmp_path / "rhs.txt"
    rhs_path.write_text("1.0\n2.0\n", encoding="utf-8")
    settings = make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_baseline": {
                "rtol": 1.0e-6,
                "atol": 1.0e-14,
                "max_iterations": 10,
                "stopping_criterion": "residual_norm",
                "m_max": 20,
                "preconditioners": [{"name": "identity", "type": "identity"}],
            },
        }
    )
    entry = ComparisonRegistryEntry(
        id="raw",
        matrix_dataset="matrix",
        rhs_source={
            "kind": "raw_rhs",
            "path": rhs_path,
            "sample_index": 0,
            "row_kind": "standard",
        },
    )

    cfg = resolve_comparison_config(case_cfg, tmp_path, _planned(entry), settings)

    assert cfg.general.data.rhs_path is None
    assert cfg.general.data.rhs_source_kind is ComparisonRhsSourceKind.RAW_RHS
    assert cfg.general.data.rhs_source_params == {
        "path": rhs_path,
        "sample_index": 0,
        "row_kind": RowKind.STANDARD,
        "scale": 1.0,
        "mask": {"kind": "none"},
    }


def test_resolve_comparison_config_binds_a_dataset_rhs_to_the_planned_row(
    tmp_path: Path,
) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    settings = make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_baseline": {"preconditioners": [{"name": "identity", "type": "identity"}]},
        }
    )
    entry = ComparisonRegistryEntry(
        id="dataset", matrix_dataset="matrix", rhs_source={"kind": "dataset"}
    )

    cfg = resolve_comparison_config(case_cfg, tmp_path, _planned(entry, rhs_row=2), settings)

    assert cfg.general.data.rhs_path is None
    assert cfg.general.data.rhs_source_kind is ComparisonRhsSourceKind.DATASET
    assert cfg.general.data.rhs_source_params == {"sample_index": 2, "mask": {"kind": "none"}}


def test_comparison_data_model_tracks_explicit_normalize_in_fields_set() -> None:
    """model_fields_set must distinguish explicit normalize_system from default."""
    from neuralls.platform.config.models.comparison import ComparisonDataModel

    default_model = ComparisonDataModel()
    assert "normalize_system" not in default_model.model_fields_set

    explicit_model = ComparisonDataModel(normalize_system="matrix")
    assert "normalize_system" in explicit_model.model_fields_set


def test_validate_neural_preconditioner_requires_model_ref() -> None:
    spec = NeuralPreconditionerConfig(type=PreconditionerType.NEURAL)
    with pytest.raises(ValueError):
        _validate_neural_preconditioner(spec)


def test_validate_neural_preconditioner_rejects_checkpoint_path() -> None:
    spec = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        checkpoint_path=Path("/workspace/model.ckpt"),
        model_ref=_logged_ref("run-1"),
    )
    with pytest.raises(ValueError):
        _validate_neural_preconditioner(spec)


def test_resolve_neural_preconditioners_validates_all() -> None:
    specs = [
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        NeuralPreconditionerConfig(
            type=PreconditionerType.NEURAL,
            model_ref=_logged_ref("run-1"),
        ),
    ]
    resolved = _resolve_neural_preconditioners(specs)
    assert len(resolved) == 2


def test_neural_specs_from_assignments_uses_strict_assignment_ref(
    tmp_path: Path,
) -> None:
    """A train-kind assignment yields an unresolved neural preconditioner stub."""
    cfg, config_dir, settings = _case_config_with_assignment(
        tmp_path, job_writer=_write_train_job_config
    )
    entry = AssignmentEntry(
        id="ffnn_solutions",
        dataset="solutions",
        job="ffnn",
        display_name="Shared model label",
    )
    client = MagicMock()
    client.search_experiments.return_value = []

    specs = neural_specs_from_assignments(
        [entry], claimed_ids=set(), client=client, cfg=cfg, config_dir=config_dir, settings=settings
    )

    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, NeuralPreconditionerConfig)
    assert spec.name == "neural_assignment_ffnn_solutions"
    assert spec.assignment == "ffnn_solutions"
    assert spec.model_ref == TrainedAssignmentRefConfig(assignment_id="ffnn_solutions")


def test_neural_specs_from_assignments_skips_claimed_ids(tmp_path: Path) -> None:
    """Assignments already covered by an explicit preconditioner are skipped."""
    cfg, config_dir, settings = _case_config_with_assignment(
        tmp_path, job_writer=_write_train_job_config
    )
    entry = AssignmentEntry(id="ffnn_solutions", dataset="solutions", job="ffnn")
    client = MagicMock()

    specs = neural_specs_from_assignments(
        [entry],
        claimed_ids={"ffnn_solutions"},
        client=client,
        cfg=cfg,
        config_dir=config_dir,
        settings=settings,
    )

    assert specs == []


def test_neural_specs_from_assignments_dispatches_fit_kind_to_pod_stub(
    tmp_path: Path,
) -> None:
    """A `run.type = "fit"` assignment yields an unresolved AMG/POD stub, not neural."""
    cfg, config_dir, settings = _case_config_with_assignment(
        tmp_path,
        job_writer=lambda path: _write_fit_job_config(path, rank=0.9999),
        assignment_id="pod2g_cg1",
        dataset_id="solutions-cg1",
        job_id="pod-2g_cg-0",
    )
    entry = AssignmentEntry(
        id="pod2g_cg1",
        dataset="solutions-cg1",
        job="pod-2g_cg-0",
        display_name="Shared model label",
    )
    client = MagicMock()

    specs = neural_specs_from_assignments(
        [entry], claimed_ids=set(), client=client, cfg=cfg, config_dir=config_dir, settings=settings
    )

    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, AMGPreconditionerConfig)
    assert "pod_rank0p9999_n-1_raw_assignment_pod2g_cg1" in spec.name
    assert isinstance(spec.coarsening, PODCoarseningConfig)
    assert spec.coarsening.assignment == "pod2g_cg1"
    assert spec.coarsening.rank == 0.9999
    assert spec.coarsening.dataset_dir == settings.processed_dir / "solutions-cg1"
    assert spec.coarsening.model_ref == TrainedAssignmentRefConfig(assignment_id="pod2g_cg1")
    client.search_experiments.assert_not_called()


def _build_split_pinned_dataset(root: Path) -> Path:
    """Single-matrix, 3-row dataset with a pinned split test fold of rows (1, 2)."""
    from dlkit.infrastructure.io.index import save_split_indices
    from dlkit.infrastructure.types.split import IndexSplit

    from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
    from tests.dataset_support import build_test_dataset

    root.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
    matrix_path = root / "matrix.npy"
    np.save(matrix_path, matrix)
    dataset_dir = root / "dataset"
    build_test_dataset(
        SourceSpec(matrix_path=str(matrix_path)),
        DatasetSpec(
            mixture=MixtureSpec(counts={"neutral_ones": 3}, seed=42, shuffle=False),
            normalize="none",
        ),
        str(dataset_dir),
        dataset_format="hdf5",
    )
    save_split_indices(
        IndexSplit(train=(0,), validation=(), test=(1, 2), predict=None),
        dataset_dir / "split.json",
    )
    return dataset_dir


def _executing(pending: PendingComparison) -> PreparedComparisonExecution:
    """Narrow a pending comparison to the execution the test expects it to be."""
    match pending:
        case PreparedComparisonExecution():
            return pending
        case _:
            pytest.fail(f"expected a prepared execution, got {pending!r}")


def _resolved_input_of(prepared: PreparedComparisonExecution) -> ResolvedComparisonInput:
    """Resolve the input a prepared system will solve, from its own config and planned binding."""
    data = prepared.cfg.general.data
    return resolve_comparison_input(
        matrix_provenance=prepared.planned.provenance,
        matrix_path=Path(data.matrix_path),
        matrix_dataset_id=prepared.entry.matrix_dataset,
        matrix_index=data.matrix_index,
        require_non_residual_rhs=data.require_non_residual_rhs,
        seed=data.selection_seed,
        rhs_source_kind=data.rhs_source_kind,
        rhs_source_params=data.rhs_source_params,
    )


@pytest.fixture
def settings(tmp_path: Path) -> NeurallsSettings:
    """Runtime settings rooted in the test's tmp_path."""
    return make_settings(tmp_path)


@pytest.fixture
def experiments_config(tmp_path: Path) -> Path:
    """A case config whose comparison baseline is a lone identity preconditioner."""
    path = tmp_path / "experiments.toml"
    write_experiments_config(path)
    return path


@pytest.fixture
def system_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Minimal matrix and RHS arrays on disk, as the input-source preflight reads them."""
    return _write_system_inputs(tmp_path)


@pytest.fixture
def identity_cfg(system_inputs: tuple[Path, Path]) -> MagicMock:
    """A mocked comparison config over `system_inputs` with the identity preconditioner."""
    matrix_path, rhs_path = system_inputs
    return mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)],
    )


@pytest.fixture
def split_pinned_dataset(tmp_path: Path) -> Path:
    """A single-matrix, 3-row dataset with a pinned split test fold of rows (1, 2)."""
    return _build_split_pinned_dataset(tmp_path)


@pytest.fixture
def prepare_entry(
    experiments_config: Path, settings: NeurallsSettings
) -> Iterator[Callable[[ComparisonConfig, PlannedSystem], PendingComparison]]:
    """Prepare one entry through `_prepare_comparison_entry`, releasing its leases afterwards.

    Per-system reuse is off, so preparation never queries MLflow.
    """
    with contextlib.ExitStack() as stack:

        def prepare(cfg: ComparisonConfig, planned: PlannedSystem) -> PendingComparison:
            context = _ComparisonResolutionContext(
                topology=_resolve_comparison_topology(experiments_config, settings),
                case_config_path=experiments_config,
                settings=settings,
                load_dataset=NO_DATASET_READS,
                reuse_comparisons=False,
            )
            pending = _prepare_comparison_entry(cfg, planned, context)
            if isinstance(pending, PreparedComparisonExecution):
                stack.enter_context(pending.cleanup)
            return pending

        yield prepare


def test_prepare_injects_master_topology(
    experiments_config: Path,
    identity_cfg: MagicMock,
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    write_experiments_config(experiments_config, comparison_name="CustomComparison")

    prepared = _executing(prepare_entry(identity_cfg, _planned(_make_entry())))

    assert prepared.topology.experiment_name == "CustomComparison"


def test_prepare_keeps_the_entry_indices_unchanged(
    system_inputs: tuple[Path, Path],
    identity_cfg: MagicMock,
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    """Case-driven comparison forwards the entry's explicit sample indices unchanged."""
    _, rhs_path = system_inputs
    identity_cfg.general.data.matrix_index = 4
    identity_cfg.general.data.rhs_source_params["sample_index"] = 9
    np.save(rhs_path, np.eye(10, 2, dtype=np.float64))

    prepared = _executing(prepare_entry(identity_cfg, _planned(_make_entry())))

    data = prepared.cfg.general.data
    assert data.matrix_index == 4
    assert data.rhs_source_params == {
        "path": rhs_path,
        "sample_index": 9,
        "row_kind": RowKind.STANDARD,
        "scale": 1.0,
    }


def test_prepare_binds_the_planned_dataset_row_and_planner_provenance(
    split_pinned_dataset: Path,
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    """A dataset-kind system solves on its planned held-out row, with the planner's provenance."""
    cfg = mock_cfg(
        matrix_path=split_pinned_dataset,
        rhs_path=None,
        preconditioners=[StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)],
    )
    cfg.general.data.rhs_source_kind = ComparisonRhsSourceKind.DATASET
    cfg.general.data.rhs_source_params = {"sample_index": 1}

    prepared = _executing(prepare_entry(cfg, _planned(_make_entry(), rhs_row=1)))

    resolved_input = _resolved_input_of(prepared)
    assert resolved_input.rhs_sample_index == 1
    assert resolved_input.matrix_provenance == SharedSingleMatrix(raw_matrix_index=0)


def test_prepare_attaches_the_planned_held_out_provenance(
    identity_cfg: MagicMock,
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    planned = replace(
        _planned(_make_entry()), provenance=HeldOutMatrix(matrix_id=MatrixId(9), raw_matrix_index=0)
    )

    prepared = _executing(prepare_entry(identity_cfg, planned))

    assert _resolved_input_of(prepared).matrix_provenance == HeldOutMatrix(
        matrix_id=MatrixId(9), raw_matrix_index=0
    )


def test_prepare_generated_rhs_accepts_matrix_only_input(
    system_inputs: tuple[Path, Path],
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    """Generated-RHS comparisons should prevalidate only the matrix source."""
    matrix_path, _ = system_inputs
    cfg = mock_cfg(
        matrix_path=matrix_path,
        rhs_path=None,
        preconditioners=[StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)],
    )
    cfg.general.data.rhs_source_kind = ComparisonRhsSourceKind.GAUSSIAN
    cfg.general.data.rhs_source_params = {"mean": 0.0, "std": 1.0}
    cfg.general.data.selection_seed = 5

    prepared = _executing(prepare_entry(cfg, _planned(_make_entry())))

    assert prepared.cfg.general.data.rhs_path is None


def test_resolved_generated_rhs_uses_matrix_source_without_rhs_dataset(
    identity_cfg: MagicMock,
) -> None:
    identity_cfg.general.data.rhs_path = None
    identity_cfg.general.data.matrix_index = 0
    identity_cfg.general.data.require_non_residual_rhs = True
    identity_cfg.general.data.selection_seed = None
    identity_cfg.general.data.rhs_source_kind = ComparisonRhsSourceKind.GAUSSIAN
    identity_cfg.general.data.rhs_source_params = {
        "mask": {"kind": "random", "density": 0.5},
    }
    entry = ComparisonRegistryEntry(
        id="generated-masked",
        matrix_dataset="solutions",
        rhs_source={"kind": "gaussian", "mask": {"kind": "random", "density": 0.5}},
    )

    resolved = resolve_comparison_input(
        matrix_provenance=SharedSingleMatrix(raw_matrix_index=0),
        matrix_path=Path(identity_cfg.general.data.matrix_path),
        matrix_dataset_id=entry.matrix_dataset,
        matrix_index=identity_cfg.general.data.matrix_index,
        require_non_residual_rhs=identity_cfg.general.data.require_non_residual_rhs,
        seed=identity_cfg.general.data.selection_seed,
        rhs_source_kind=identity_cfg.general.data.rhs_source_kind,
        rhs_source_params=identity_cfg.general.data.rhs_source_params,
    )

    assert resolved.rhs_source_type == "generated"
    assert resolved.rhs_dataset_id is None
    assert resolved.rhs_sample_index is None
    assert resolved.rhs_source_kind is ComparisonRhsSourceKind.GAUSSIAN
    assert resolved.rhs_source_params == {"mask": {"kind": "random", "density": 0.5}}
    assert np.count_nonzero(resolved.rhs) == 1


def test_prepare_does_not_require_split_artifacts(
    identity_cfg: MagicMock,
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    """Case-driven comparison should not inspect DLKit split artifacts."""
    with patch(
        "neuralls.platform.tracking.splits.download_split_indices",
        autospec=True,
        side_effect=AssertionError("split artifacts should not be consulted"),
    ):
        pending = prepare_entry(identity_cfg, _planned(_make_entry()))

    assert isinstance(pending, PreparedComparisonExecution)


def test_sanitize_mlflow_metric_segment_replaces_invalid_characters() -> None:
    assert (
        sanitize_metric_key_segment(
            "Residual-Error 50 Gaussian 93x31 | Scale-Equivariant Constant-Width 3000"
        )
        == "Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000"
    )


def test_prepare_warns_and_continues_when_neural_resolution_fails(
    system_inputs: tuple[Path, Path],
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    matrix_path, rhs_path = system_inputs
    cfg = mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
            NeuralPreconditionerConfig(
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref("MissingFFNN", "solutions"),
            ),
        ],
    )
    skipped = (
        "Skipping neural preconditioner 'missing-neural': Registered model 'MissingFFNN' not found",
    )

    with patch(
        _RESOLVE_MODELS,
        return_value=MagicMock(
            specs=[StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)],
            warnings=skipped,
        ),
    ):
        prepared = _executing(prepare_entry(cfg, _planned(_make_entry())))

    assert prepared.resolved.warnings == skipped
    assert [spec.name for spec in prepared.resolved.specs] == ["identity"]


def test_prepare_fails_if_all_preconditioners_are_skipped(
    system_inputs: tuple[Path, Path],
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    matrix_path, rhs_path = system_inputs
    cfg = mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            NeuralPreconditionerConfig(
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref("MissingFFNN", "solutions"),
            ),
        ],
    )

    with patch(
        _RESOLVE_MODELS,
        return_value=MagicMock(
            specs=[],
            warnings=(
                "Skipping neural preconditioner 'missing-neural': Registered model 'MissingFFNN' not found",
            ),
        ),
    ):
        pending = prepare_entry(cfg, _planned(_make_entry()))

    assert isinstance(pending, ComparisonEntryFailed)
    assert pending.reason == NO_RUNNABLE_PRECONDITIONERS


def test_prepare_ignores_unrelated_broken_experiments(
    tmp_path: Path,
    system_inputs: tuple[Path, Path],
    experiments_config: Path,
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    (tmp_path / "datasets").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "jobs").mkdir()
    (tmp_path / "datasets" / "valid-dataset.toml").write_text(
        'id = "valid-dataset"\n',
        encoding="utf-8",
    )
    (tmp_path / "models" / "valid-model.toml").write_text(
        "[model]\nname = 'NormScaledLinearFFNN'\nmodule_path = 'dlkit.nn'\n\n[data]\nname = 'FlexibleDataset'\n\n[data.module]\nname = 'ArrayDataModule'",
        encoding="utf-8",
    )
    (tmp_path / "jobs" / "valid-job.toml").write_text(
        '[run]\ntype = "train"\nseed = 42\nmodel = "../models/valid-model.toml"\ndata = "../models/valid-model.toml"\n\n[training.trainer]\nmax_epochs = 1\n\n[training.optimizer.default_optimizer]\nname = "AdamW"\nlr = 1e-3',
        encoding="utf-8",
    )
    experiments_config.write_text(
        f"""
[mlflow]
tracking_uri = "{build_sqlite_tracking_uri(tmp_path / "mlflow.db")}"

[[datasets]]
id = "valid-dataset"
path = "datasets/valid-dataset.toml"

[[jobs]]
id = "valid-job"
path = "jobs/valid-job.toml"

[[assignments]]
id = "valid-exp"
dataset = "valid-dataset"
job = "valid-job"
        """.strip(),
        encoding="utf-8",
    )
    matrix_path, rhs_path = system_inputs
    cfg = mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            NeuralPreconditionerConfig(
                type=PreconditionerType.NEURAL,
                architecture="Linear Net",
                model_ref=_registered_ref("NormScaledLinearFFNN", "residuals-100"),
            ),
        ],
    )

    with patch(
        _RESOLVE_MODELS,
        return_value=MagicMock(specs=cfg.preconditioners, warnings=()),
    ) as mock_resolve_specs:
        pending = prepare_entry(cfg, _planned(_make_entry()))

    assert isinstance(pending, PreparedComparisonExecution)
    assert mock_resolve_specs.call_args.kwargs["assignment_contexts"] is None


def test_prepare_fails_when_matrix_input_is_missing(
    tmp_path: Path,
    system_inputs: tuple[Path, Path],
    fake_mlflow: MagicMock,
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    _, rhs_path = system_inputs
    cfg = mock_cfg(
        matrix_path=tmp_path / "missing-matrix.npy",
        rhs_path=rhs_path,
        preconditioners=[StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)],
    )

    pending = prepare_entry(cfg, _planned(_make_entry()))

    assert isinstance(pending, ComparisonEntryFailed)
    fake_mlflow.start_run.assert_not_called()


def test_prepare_fails_when_a_known_dataset_is_missing(
    tmp_path: Path,
    system_inputs: tuple[Path, Path],
    prepare_entry: Callable[[ComparisonConfig, PlannedSystem], PendingComparison],
) -> None:
    _, rhs_path = system_inputs
    cfg = mock_cfg(
        matrix_path=tmp_path / "solutions",
        rhs_path=rhs_path,
        preconditioners=[StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)],
    )

    pending = prepare_entry(cfg, _planned(_make_entry()))

    assert isinstance(pending, ComparisonEntryFailed)


@pytest.mark.usefixtures("stub_input_gate", "stub_planning")
def test_run_comparison_batch_preserves_declared_order(
    tmp_path: Path, fake_mlflow: MagicMock, comparison_run: NamedRun
) -> None:
    """Cache-hit outcomes come back in declared config order.

    Also proves no MLflow run is needed for this: run_comparison_batch never
    calls mlflow.start_run when _prepare_comparison_entry reports every entry
    as a cache hit.
    """
    experiments_config = tmp_path / "experiments.toml"
    write_experiments_config(experiments_config, with_comparisons=True)

    def _fake_prepare_entry(
        cfg: object,
        planned: PlannedSystem,
        context: object,
        *,
        force: bool = False,
    ) -> ComparisonOutcome:
        return ComparisonReused(comparison_id=planned.entry.id, run=comparison_run)

    with (
        patch(
            "neuralls.composition.assignments.comparison_batch._prepare_comparison_entry",
            side_effect=_fake_prepare_entry,
        ) as mock_prepare,
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_config",
            return_value=mock_cfg(),
        ),
    ):
        outcomes = run_comparison_batch(experiments_config, ComparisonParams())

    assert [outcome.comparison_id for outcome in outcomes] == ["a", "b"]
    assert [call.args[1].entry.id for call in mock_prepare.call_args_list] == ["a", "b"]
    fake_mlflow.start_run.assert_not_called()


def test_compatibility_projection_carries_canonical_setup_measurement(
    tmp_path: Path,
    prepared_method: PreparedMethodFactory,
) -> None:
    """Legacy diagnostics retain setup timing copied from the canonical event."""
    tag = SystemTag(
        dataset=DatasetId("dataset"),
        recipe=GAUSSIAN,
        matrix_id=MatrixId(3),
        rhs_ordinal=0,
        system_id=SystemId("system-3-0"),
    )
    workload = CompletedMethodWorkload(
        method="identity",
        planned=(tag,),
        setup_events=(
            MatrixSetupEvent(
                event_id=SetupEventId("identity:dataset:3"),
                dataset=tag.dataset,
                matrix_id=tag.matrix_id,
                method="identity",
                setup_s=1.25,
                samples_s=(1.0, 1.25, 1.5),
                peak_memory=MeasuredPeakMemory(128),
                outcome=SetupCompleted(),
            ),
        ),
        solve_attempts=(),
        diagnostics=(),
        generation_cost=None,
        workload_peak_memory=MeasuredPeakMemory(128),
        config_digest="sha256:" + "0" * 64,
        method_config={"name": "identity"},
        presentation=ExecutedMethod(prepared_method("identity"), ()),
    )
    projected = _project_setup_measurements(
        workload,
        str(tag.system_id),
        _typed_comparison_result(tmp_path / "conv.png").results["identity"],
    )

    assert projected.setup_cost == 1.25
    assert projected.setup_time_samples == (1.0, 1.25, 1.5)
    assert projected.setup_peak_memory_bytes == 128


@pytest.mark.usefixtures("stub_input_gate", "stub_setup_tracking", "fake_mlflow")
def test_comparison_sweep_runs_one_system_per_held_out_matrix_and_rhs_draw(
    tmp_path: Path,
    sparse_id_dataset: Path,
    pin_test_fold: Callable[[Path, tuple[int, ...]], None],
) -> None:
    """A `first_n` sweep over a multi-matrix dataset solves each selected matrix `rhs_count` times.

    Matrix ids (5, 9) differ from every storage row, so the run proves the
    runner loads each matrix by its planned row and tags it with the planner's
    held-out provenance.
    """
    settings = make_settings(tmp_path)
    dataset_dir = settings.processed_dir / "solutions"
    dataset_dir.parent.mkdir(parents=True)
    sparse_id_dataset.rename(dataset_dir)
    pin_test_fold(dataset_dir, (0, 1, 2, 3))

    experiments_config = tmp_path / "experiments.toml"
    payload: dict[str, object] = {
        "mlflow": {"tracking_uri": build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")},
        "names": {"training": "neuralls-training", "comparison": "Comparisons"},
        "comparison_baseline": {
            "rtol": 1.0e-6,
            "atol": 1.0e-14,
            "max_iterations": 10,
            "stopping_criterion": "residual_norm",
            "m_max": 20,
            "preconditioners": [{"name": "identity", "type": "identity"}],
        },
        "datasets": [{"id": "solutions", "path": "datasets/solutions.toml"}],
        "comparison_sweeps": [
            {
                "matrix_dataset": "solutions",
                "rhs_source": {"kind": "gaussian"},
                "rhs_count": 2,
                "seed": 42,
                "selector": {"kind": "first_n", "n": 2},
                "id_prefix": "sweep",
            }
        ],
    }
    (tmp_path / "datasets").mkdir(exist_ok=True)
    write_dataset_config(tmp_path / "datasets" / "solutions.toml", "solutions")
    with experiments_config.open("wb") as fh:
        tomli_w.dump(payload, fh)

    inputs: list[ResolvedComparisonInput] = []
    bundles_seen: list[MatrixWorkload] = []

    def capture_resolved_input(
        *,
        matrix_path: Path,
        matrix_dataset_id: str,
        matrix_index: int,
        matrix_provenance: MatrixProvenance,
        require_non_residual_rhs: bool,
        seed: int | None,
        rhs_source_kind: ComparisonRhsSourceKind | str,
        rhs_source_params: dict[str, object] | None,
    ) -> ResolvedComparisonInput:
        resolved = resolve_comparison_input(
            matrix_path=matrix_path,
            matrix_dataset_id=matrix_dataset_id,
            matrix_index=matrix_index,
            matrix_provenance=matrix_provenance,
            require_non_residual_rhs=require_non_residual_rhs,
            seed=seed,
            rhs_source_kind=rhs_source_kind,
            rhs_source_params=rhs_source_params,
        )
        inputs.append(resolved)
        return resolved

    def capture_workload(
        method: ResolvedMethod,
        bundles: Iterable[MatrixWorkload],
        *,
        prepared: PreparedMethodPresentation,
        datasets: Mapping[DatasetId, DatasetPresentation],
        notation: NotationPolicy,
        params: SolverParams,
        settings: NeurallsSettings,
    ) -> CompletedMethodWorkload:
        bundle_sequence = tuple(bundles)
        bundles_seen.extend(bundle_sequence)
        return execute_method_workload(
            method,
            bundle_sequence,
            prepared=prepared,
            datasets=datasets,
            notation=notation,
            params=params,
            settings=settings,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_input",
            side_effect=capture_resolved_input,
        ),
        patch(
            "neuralls.composition.assignments.comparison_batch.execute_method_workload",
            side_effect=capture_workload,
        ),
    ):
        outcomes = run_comparison_batch(experiments_config, ComparisonParams(), settings)

    assert all(isinstance(outcome, ComparisonCompleted) for outcome in outcomes)
    assert [outcome.comparison_id for outcome in outcomes] == [
        "sweep-5-0",
        "sweep-5-1",
        "sweep-9-0",
        "sweep-9-1",
    ]
    expected_provenance = [
        HeldOutMatrix(matrix_id=MatrixId(5), raw_matrix_index=0),
        HeldOutMatrix(matrix_id=MatrixId(5), raw_matrix_index=0),
        HeldOutMatrix(matrix_id=MatrixId(9), raw_matrix_index=2),
        HeldOutMatrix(matrix_id=MatrixId(9), raw_matrix_index=2),
    ]
    assert [i.matrix_provenance for i in inputs] == expected_provenance * 2
    assert [float(i.matrix[0, 0]) for i in inputs] == [5.0, 5.0, 9.0, 9.0] * 2
    # The comparison config's matrix normalization is applied while staging
    # each bundle, so the diagonal 2x2 matrices have spectral norm 1/sqrt(2).
    assert [
        float(np.linalg.norm(bundle.matrix, ord=2)) for bundle in bundles_seen
    ] == pytest.approx([1 / np.sqrt(2), 1 / np.sqrt(2)])
    # Each system draws its own RHS, so no two vectors coincide.
    assert len({tuple(i.rhs) for i in inputs}) == 4


@pytest.mark.usefixtures("stub_input_gate", "stub_setup_tracking", "fake_mlflow")
def test_an_external_rhs_of_the_wrong_length_fails_only_its_own_system(
    tmp_path: Path,
    single_matrix_dataset: Path,
    pin_test_fold: Callable[[Path, tuple[int, ...]], None],
) -> None:
    """A raw RHS that does not fit the native 2x2 matrix fails its system; the batch goes on."""
    settings = make_settings(tmp_path)
    dataset_dir = settings.processed_dir / "solutions"
    dataset_dir.parent.mkdir(parents=True)
    single_matrix_dataset.rename(dataset_dir)
    pin_test_fold(dataset_dir, (0, 1))
    wrong_length_rhs = tmp_path / "rhs-of-length-3.npy"
    np.save(wrong_length_rhs, np.ones(3))

    experiments_config = tmp_path / "experiments.toml"
    payload: dict[str, object] = {
        "mlflow": {"tracking_uri": build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")},
        "names": {"training": "neuralls-training", "comparison": "Comparisons"},
        "comparison_baseline": {
            "rtol": 1.0e-6,
            "atol": 1.0e-14,
            "max_iterations": 10,
            "stopping_criterion": "residual_norm",
            "m_max": 20,
            "preconditioners": [{"name": "identity", "type": "identity"}],
        },
        "datasets": [{"id": "solutions", "path": "datasets/solutions.toml"}],
        "comparisons": [
            {"id": "fits", "matrix_dataset": "solutions", "rhs_source": {"kind": "gaussian"}},
            {
                "id": "too-long",
                "matrix_dataset": "solutions",
                "rhs_source": {
                    "kind": "raw_rhs",
                    "path": str(wrong_length_rhs),
                    "row_kind": "standard",
                },
            },
        ],
    }
    (tmp_path / "datasets").mkdir(exist_ok=True)
    write_dataset_config(tmp_path / "datasets" / "solutions.toml", "solutions")
    with experiments_config.open("wb") as fh:
        tomli_w.dump(payload, fh)

    outcomes = run_comparison_batch(experiments_config, ComparisonParams(), settings)

    by_id = {outcome.comparison_id: outcome for outcome in outcomes}
    assert isinstance(by_id["fits"], ComparisonCompleted)
    assert isinstance(by_id["too-long"], ComparisonEntryFailed)
