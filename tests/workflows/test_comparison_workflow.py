"""Tests for neuralls.composition.assignments.comparison_batch."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tomli_w

from neuralls.composition.assignments.comparison_batch import (
    _resolve_comparison_topology,
    _resolve_neural_preconditioners,
    _run_comparison_from_config,
    _validate_neural_preconditioner,
    neural_specs_from_assignments,
    run_comparison_batch,
)
from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.comparison.config_assembler import resolve_comparison_config
from neuralls.composition.comparison.models import ComparisonOutcome, ComparisonParams
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonRecommendations,
    ComparisonResult,
    PlotPaths,
    RankedRecommendation,
)
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
    PreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.reporting.artifacts import (
    extract_array_artifacts,
    serialize_comparison_payload,
)
from neuralls.platform.tracking.comparison_tracking import (
    log_comparison_result_metrics,
    log_linear_system_params,
)
from neuralls.platform.tracking.mlflow import sanitize_metric_key_segment
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind


def test_resolve_comparison_config_importable_from_composition() -> None:
    """resolve_comparison_config must live in composition, not platform."""
    from neuralls.composition.comparison.config_assembler import (
        resolve_comparison_config,  # noqa: F401
    )


_RESOLVE_COMPARISON_CONFIG = (
    "neuralls.composition.comparison.config_assembler.resolve_comparison_config"
)
_COMPARE_PRECONDITIONERS = (
    "neuralls.composition.assignments.comparison_batch.compare_preconditioners"
)
_MLFLOW_MODULE = "neuralls.composition.assignments.comparison_batch.mlflow"
_COMPARISON_TRACKING_MLFLOW_MODULE = "neuralls.platform.tracking.comparison_tracking.mlflow"
_SETUP_TRACKING = "neuralls.composition.assignments.comparison_batch.setup_comparison_tracking"
_LOG_COMPARISON_ARTIFACTS = (
    "neuralls.composition.assignments.comparison_batch.log_comparison_artifacts_to_mlflow"
)


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
                'name = "none"',
                'type = "identity"',
            ]
        ),
        encoding="utf-8",
    )


def _write_experiments_config(
    path: Path,
    *,
    with_comparisons: bool = False,
    comparison_name: str = "Comparisons",
) -> None:
    payload: dict[str, object] = {
        "mlflow": {"tracking_uri": build_sqlite_tracking_uri(path.parent / "mlruns" / "mlflow.db")},
        "names": {
            "training": "neuralls-training",
            "comparison": comparison_name,
        },
        "comparison_defaults": {
            "rtol": 1.0e-6,
            "atol": 1.0e-14,
            "max_iterations": 10,
            "stopping_criterion": "residual_norm",
            "m_max": 20,
            "preconditioners": [{"name": "none", "type": "identity"}],
        },
    }
    if with_comparisons:
        (path.parent / "datasets").mkdir(exist_ok=True)
        _write_dataset_config(path.parent / "datasets" / "solutions.toml", "solutions")
        _write_dataset_config(path.parent / "datasets" / "gaussian-rhs.toml", "gaussian-rhs")
        payload["datasets"] = [
            {"id": "solutions", "path": "datasets/solutions.toml"},
            {"id": "gaussian-rhs", "path": "datasets/gaussian-rhs.toml"},
        ]
        payload["comparisons"] = [
            {"id": "a", "matrix_dataset": "solutions", "rhs_source": {"kind": "gaussian"}},
            {"id": "b", "matrix_dataset": "solutions", "rhs_source": {"kind": "gaussian"}},
        ]
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)


def _write_dataset_config(path: Path, dataset_id: str) -> None:
    """Write a minimal dataset config stub for registry resolution."""
    path.write_text(
        "\n".join(
            [
                f'id = "{dataset_id}"',
                "",
                "[output]",
                "",
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
    _write_dataset_config(datasets_dir / f"{dataset_id}.toml", dataset_id)

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    job_writer(jobs_dir / f"{job_id}.toml")

    settings = _make_settings(tmp_path)
    cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": dataset_id, "path": f"datasets/{dataset_id}.toml"}],
            "jobs": [{"id": job_id, "path": f"jobs/{job_id}.toml"}],
            "assignments": [{"id": assignment_id, "dataset": dataset_id, "job": job_id}],
        }
    )
    return cfg, tmp_path, settings


def _configure_mock_mlflow(mock_mlflow: MagicMock, run_id: str = "comp-run-id") -> None:
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_run.return_value.__exit__.return_value = False
    mock_mlflow.get_artifact_uri.return_value = f"mlartifacts/0/{run_id}/artifacts"


def _mock_cfg(
    *,
    matrix_path: Path | None = None,
    rhs_path: Path | None = None,
    preconditioners: list[PreconditionerConfig] | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.general.data.matrix_path = matrix_path
    cfg.general.data.rhs_path = rhs_path
    cfg.general.data.matrix_index = 0
    cfg.general.data.dataset_alias = None
    cfg.general.data.normalize_system = "matrix"
    cfg.general.data.require_non_residual_rhs = True
    cfg.general.data.selection_seed = None
    cfg.general.data.rhs_source_kind = (
        ComparisonRhsSourceKind.RAW_RHS
        if rhs_path is not None
        else ComparisonRhsSourceKind.GAUSSIAN
    )
    cfg.general.data.rhs_source_params = (
        {"path": rhs_path, "sample_index": 0, "row_kind": RowKind.STANDARD, "scale": 1.0}
        if rhs_path is not None
        else {"mean": 0.0, "std": 1.0}
    )
    cfg.preconditioners = tuple(preconditioners or [])
    return cfg


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
            "none": CGComparisonResult(
                x=np.array([1.0, 2.0]),
                converged=True,
                iterations=2,
                residual=1.0e-8,
                residual_abs=1.0e-9,
                residual_history_rel=[1.0, 1.0e-8],
                residual_history_abs=[1.0, 1.0e-9],
                preconditioner="none",
                initial_guess=np.zeros(2),
                exact_error=None,
                rhs_norm=1.0,
                breakdown=False,
            )
        },
        summary="ok",
        solver_params=_solver_params(plot_path.parent),
        plot_paths=PlotPaths(convergence=plot_path),
        preconditioners=("none",),
        condition_numbers={"none": 1.0},
        recommendations=ComparisonRecommendations(
            ranked=(
                RankedRecommendation(
                    label="none",
                    iterations=2,
                    residual=1.0e-8,
                    residual_abs=1.0e-9,
                    breakdown=False,
                ),
            ),
            overall_best=RankedRecommendation(
                label="none",
                iterations=2,
                residual=1.0e-8,
                residual_abs=1.0e-9,
                breakdown=False,
            ),
        ),
    )


@pytest.fixture
def stub_comparison_result(tmp_path: Path) -> ComparisonResult:
    """An empty-but-real result standing in for a patched `compare_preconditioners`.

    The artifact writer consumes typed `ComparisonResult`s only — a MagicMock
    would not be a valid substitute for the real workflow's return value.
    """
    return ComparisonResult(
        results={},
        summary="",
        solver_params=_solver_params(tmp_path),
        preconditioners=(),
        condition_numbers={},
        recommendations=ComparisonRecommendations(),
    )


def _make_entry(comparison_id: str = "test") -> ComparisonRegistryEntry:
    """Build a minimal ComparisonRegistryEntry for direct _run_comparison_from_config calls."""
    return ComparisonRegistryEntry(
        id=comparison_id,
        matrix_dataset="solutions",
        rhs_source={"kind": "gaussian"},
    )


def _make_settings(tmp_path: Path) -> NeurallsSettings:
    """Build a NeurallsSettings backed by tmp_path."""
    return NeurallsSettings(
        _env_file=[],
        processed_dir=tmp_path / "processed",
        output_dir=tmp_path / "output",
    )


def test_resolve_comparison_config_recovers_omitted_index_semantics(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    _write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    _write_dataset_config(datasets_dir / "rhs.toml", "rhs")
    settings = _make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [
                {"id": "matrix", "path": "datasets/matrix.toml"},
                {"id": "rhs", "path": "datasets/rhs.toml"},
            ],
            "comparison_defaults": {
                "rtol": 1.0e-6,
                "atol": 1.0e-14,
                "max_iterations": 10,
                "stopping_criterion": "residual_norm",
                "m_max": 20,
                "preconditioners": [{"name": "none", "type": "identity"}],
            },
        }
    )
    omitted_entry = ComparisonRegistryEntry(
        id="omitted",
        matrix_dataset="matrix",
        rhs_source={
            "kind": "dataset",
            "path": settings.processed_dir / "rhs",
        },
    )
    explicit_entry = ComparisonRegistryEntry(
        id="explicit",
        matrix_dataset="matrix",
        matrix_index=0,
        rhs_source={
            "kind": "dataset",
            "path": settings.processed_dir / "rhs",
            "sample_index": 0,
        },
    )

    omitted_cfg = resolve_comparison_config(case_cfg, tmp_path, omitted_entry, settings)
    explicit_cfg = resolve_comparison_config(case_cfg, tmp_path, explicit_entry, settings)

    assert omitted_cfg.general.data.matrix_index is None
    assert explicit_cfg.general.data.matrix_index == 0
    assert explicit_cfg.general.data.rhs_source_params is not None
    assert explicit_cfg.general.data.rhs_source_params["sample_index"] == 0


def test_resolve_comparison_config_carries_generated_rhs_without_rhs_path(
    tmp_path: Path,
) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    _write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    settings = _make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_defaults": {
                "rtol": 1.0e-6,
                "atol": 1.0e-14,
                "max_iterations": 10,
                "stopping_criterion": "residual_norm",
                "m_max": 20,
                "preconditioners": [{"name": "none", "type": "identity"}],
            },
        }
    )
    entry = ComparisonRegistryEntry(
        id="generated",
        matrix_dataset="matrix",
        rhs_source={"kind": "gaussian"},
        seed=42,
    )

    cfg = resolve_comparison_config(case_cfg, tmp_path, entry, settings)

    assert cfg.general.data.rhs_path is None
    assert cfg.general.data.rhs_source_kind is ComparisonRhsSourceKind.GAUSSIAN
    assert cfg.general.data.selection_seed == 42


def test_resolve_comparison_config_carries_raw_rhs_source(
    tmp_path: Path,
) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    _write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    rhs_path = tmp_path / "rhs.txt"
    rhs_path.write_text("1.0\n2.0\n", encoding="utf-8")
    settings = _make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_defaults": {
                "rtol": 1.0e-6,
                "atol": 1.0e-14,
                "max_iterations": 10,
                "stopping_criterion": "residual_norm",
                "m_max": 20,
                "preconditioners": [{"name": "none", "type": "identity"}],
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

    cfg = resolve_comparison_config(case_cfg, tmp_path, entry, settings)

    assert cfg.general.data.rhs_path is None
    assert cfg.general.data.rhs_source_kind is ComparisonRhsSourceKind.RAW_RHS
    assert cfg.general.data.rhs_source_params == {
        "path": rhs_path,
        "sample_index": 0,
        "row_kind": RowKind.STANDARD,
        "scale": 1.0,
    }


def test_resolve_comparison_config_carries_dataset_rhs_source(
    tmp_path: Path,
) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    _write_dataset_config(datasets_dir / "matrix.toml", "matrix")
    rhs_dataset_dir = tmp_path / "processed" / "rhs-dataset"
    settings = _make_settings(tmp_path)
    case_cfg = CaseConfig.model_validate(
        {
            "datasets": [{"id": "matrix", "path": "datasets/matrix.toml"}],
            "comparison_defaults": {
                "rtol": 1.0e-6,
                "atol": 1.0e-14,
                "max_iterations": 10,
                "stopping_criterion": "residual_norm",
                "m_max": 20,
                "preconditioners": [{"name": "none", "type": "identity"}],
            },
        }
    )
    entry = ComparisonRegistryEntry(
        id="dataset",
        matrix_dataset="matrix",
        rhs_source={
            "kind": "dataset",
            "path": rhs_dataset_dir,
            "sample_index": 2,
        },
    )

    cfg = resolve_comparison_config(case_cfg, tmp_path, entry, settings)

    assert cfg.general.data.rhs_path is None
    assert cfg.general.data.rhs_source_kind is ComparisonRhsSourceKind.DATASET
    assert cfg.general.data.rhs_source_params == {
        "path": rhs_dataset_dir,
        "sample_index": 2,
    }


def test_comparison_data_model_tracks_explicit_normalize_in_fields_set() -> None:
    """model_fields_set must distinguish explicit normalize_system from default."""
    from neuralls.platform.config.models.comparison import ComparisonDataModel

    default_model = ComparisonDataModel()
    assert "normalize_system" not in default_model.model_fields_set

    explicit_model = ComparisonDataModel(normalize_system="matrix")
    assert "normalize_system" in explicit_model.model_fields_set


def test_validate_neural_preconditioner_requires_model_ref() -> None:
    spec = NeuralPreconditionerConfig(name="neural", type=PreconditionerType.NEURAL)
    try:
        _validate_neural_preconditioner(spec)
    except ValueError as exc:
        assert "model_ref" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_validate_neural_preconditioner_rejects_checkpoint_path(tmp_path: Path) -> None:
    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=tmp_path / "model.ckpt",
        model_ref=_logged_ref("run-1"),
    )
    try:
        _validate_neural_preconditioner(spec)
    except ValueError as exc:
        assert "checkpoint_path/assignment" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_resolve_neural_preconditioners_validates_all() -> None:
    specs = [
        StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY),
        NeuralPreconditionerConfig(
            name="neural",
            type=PreconditionerType.NEURAL,
            model_ref=_logged_ref("run-1"),
        ),
    ]
    resolved = _resolve_neural_preconditioners(specs)
    assert len(resolved) == 2


def test_neural_specs_from_assignments_uses_logged_ref_with_assignment_tag(
    tmp_path: Path,
) -> None:
    """A train-kind assignment yields an unresolved neural preconditioner stub."""
    cfg, config_dir, settings = _case_config_with_assignment(
        tmp_path, job_writer=_write_train_job_config
    )
    entry = AssignmentEntry(id="ffnn_solutions", dataset="solutions", job="ffnn")
    client = MagicMock()
    client.search_experiments.return_value = []

    specs = neural_specs_from_assignments(
        [entry], claimed_ids=set(), client=client, cfg=cfg, config_dir=config_dir, settings=settings
    )

    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, NeuralPreconditionerConfig)
    assert spec.assignment == "ffnn_solutions"
    assert spec.model_ref == LoggedModelRefConfig(
        latest=True, tags={"assignment_id": "ffnn_solutions"}
    )


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
        job_id="pod-2g_0-cg",
    )
    entry = AssignmentEntry(id="pod2g_cg1", dataset="solutions-cg1", job="pod-2g_0-cg")
    client = MagicMock()

    specs = neural_specs_from_assignments(
        [entry], claimed_ids=set(), client=client, cfg=cfg, config_dir=config_dir, settings=settings
    )

    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, AMGPreconditionerConfig)
    assert isinstance(spec.coarsening, PODCoarseningConfig)
    assert spec.coarsening.assignment == "pod2g_cg1"
    assert spec.coarsening.rank == 0.9999
    assert spec.coarsening.dataset_dir == settings.processed_dir / "solutions-cg1"
    assert spec.coarsening.model_ref == LoggedModelRefConfig(
        latest=True, tags={"assignment_id": "pod2g_cg1"}
    )
    client.search_experiments.assert_not_called()


def test_extract_array_artifacts_detaches_numpy_data(tmp_path: Path) -> None:
    payload, array_artifacts = extract_array_artifacts(
        _typed_comparison_result(tmp_path / "convergence.png")
    )
    serialized = serialize_comparison_payload(payload)
    assert serialized["results"]["none"]["iterations"] == 2
    assert serialized["results"]["none"]["residual"] == 1.0e-8
    assert serialized["results"]["none"]["x"] == {
        "path": "arrays/results/none/x.npy",
        "shape": [2],
        "dtype": "float64",
    }
    artifact_paths = {artifact.reference.path for artifact in array_artifacts}
    assert Path("arrays/results/none/x.npy") in artifact_paths


def test_run_comparison_injects_master_topology(
    tmp_path: Path, stub_comparison_result: ComparisonResult
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config, comparison_name="CustomComparison")
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=stub_comparison_result),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_LOG_COMPARISON_ARTIFACTS),
    ):
        _configure_mock_mlflow(mock_mlflow)
        topology = _resolve_comparison_topology(experiments_config, settings)
        outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is True
    assert topology.experiment_name == "CustomComparison"


def test_run_comparison_uses_explicit_entry_indices(
    tmp_path: Path, stub_comparison_result: ComparisonResult
) -> None:
    """Case-driven comparison forwards the entry's explicit sample indices unchanged."""
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    cfg.general.data.matrix_index = 4
    cfg.general.data.rhs_source_params["sample_index"] = 9
    np.save(rhs_path, np.eye(10, 2, dtype=np.float64))
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=stub_comparison_result) as mock_compare,
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
    ):
        _configure_mock_mlflow(mock_mlflow)
        topology = _resolve_comparison_topology(experiments_config, settings)
        _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert mock_compare.call_args.kwargs["general_params"].data.matrix_index == 4
    assert (
        mock_compare.call_args.kwargs["general_params"].data.rhs_source_params["sample_index"] == 9
    )


def test_run_comparison_generated_rhs_accepts_matrix_only_input(
    tmp_path: Path, stub_comparison_result: ComparisonResult
) -> None:
    """Generated-RHS comparisons should prevalidate only the matrix source."""
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    matrix_path, _rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=None,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    cfg.general.data.rhs_source_kind = ComparisonRhsSourceKind.GAUSSIAN
    cfg.general.data.rhs_source_params = {"mean": 0.0, "std": 1.0}
    cfg.general.data.selection_seed = 5
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=stub_comparison_result),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
    ):
        _configure_mock_mlflow(mock_mlflow)
        topology = _resolve_comparison_topology(experiments_config, settings)
        outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is True


def test_resolved_generated_rhs_uses_matrix_source_without_rhs_dataset(tmp_path: Path) -> None:
    matrix_path, _rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=None,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    cfg.general.data.matrix_index = 0
    cfg.general.data.require_non_residual_rhs = True
    cfg.general.data.selection_seed = None
    cfg.general.data.rhs_source_kind = ComparisonRhsSourceKind.SPARSE
    cfg.general.data.rhs_source_params = {
        "indices": [0, 1],
        "values": [3.0, -2.0],
    }
    entry = ComparisonRegistryEntry(
        id="generated-sparse",
        matrix_dataset="solutions",
        rhs_source={
            "kind": "sparse",
            "indices": [0, 1],
            "values": [3.0, -2.0],
        },
    )

    resolved = resolve_comparison_input(
        matrix_path=Path(cfg.general.data.matrix_path),
        matrix_dataset_id=entry.matrix_dataset,
        matrix_index=cfg.general.data.matrix_index,
        require_non_residual_rhs=cfg.general.data.require_non_residual_rhs,
        seed=cfg.general.data.selection_seed,
        rhs_source_kind=cfg.general.data.rhs_source_kind,
        rhs_source_params=cfg.general.data.rhs_source_params,
    )

    assert resolved.rhs_source_type == "generated"
    assert resolved.rhs_dataset_id is None
    assert resolved.rhs_sample_index is None
    assert resolved.rhs_source_kind is ComparisonRhsSourceKind.SPARSE
    assert resolved.rhs_source_params == {"indices": [0, 1], "values": [3.0, -2.0]}
    np.testing.assert_allclose(resolved.rhs, np.array([3.0, -2.0], dtype=np.float64))


def test_run_comparison_does_not_require_split_artifacts(
    tmp_path: Path, stub_comparison_result: ComparisonResult
) -> None:
    """Case-driven comparison should not inspect DLKit split artifacts."""
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=stub_comparison_result),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
        patch(
            "neuralls.platform.tracking.splits.download_split_indices",
            side_effect=AssertionError("split artifacts should not be consulted"),
        ),
    ):
        _configure_mock_mlflow(mock_mlflow)
        topology = _resolve_comparison_topology(experiments_config, settings)
        outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is True


def test_sanitize_mlflow_metric_segment_replaces_invalid_characters() -> None:
    assert (
        sanitize_metric_key_segment(
            "Residual-Error 50 Gaussian 93x31 | Scale-Equivariant Constant-Width 3000"
        )
        == "Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000"
    )


def test_log_comparison_metrics_uses_mlflow_safe_metric_names(tmp_path: Path) -> None:
    result = ComparisonResult(
        results={
            "Residual-Error 50 Gaussian 93x31 | Scale-Equivariant Constant-Width 3000": (
                CGComparisonResult(
                    x=np.array([1.0, 2.0]),
                    converged=True,
                    iterations=2,
                    residual=1.0e-8,
                    residual_abs=1.0e-9,
                    residual_history_rel=[1.0, 1.0e-8],
                    residual_history_abs=[1.0, 1.0e-9],
                    preconditioner="neural",
                    initial_guess=np.zeros(2),
                    exact_error=None,
                    rhs_norm=1.0,
                    breakdown=False,
                )
            )
        },
        summary="ok",
        solver_params=_solver_params(tmp_path),
        preconditioners=("neural",),
        condition_numbers={
            "Residual-Error 50 Gaussian 93x31 | Scale-Equivariant Constant-Width 3000": 1.0
        },
        recommendations=ComparisonRecommendations(),
    )

    with patch(_COMPARISON_TRACKING_MLFLOW_MODULE) as mock_mlflow:
        log_comparison_result_metrics(
            result,
            child_run_tags={
                "Residual-Error 50 Gaussian 93x31 | Scale-Equivariant Constant-Width 3000": {}
            },
        )

    logged_names = [call.args[0] for call in mock_mlflow.log_metric.call_args_list[:4]]
    assert logged_names == [
        "condition_number/Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000",
        "iterations/Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000",
        "final_residual/Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000",
        "converged/Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000",
    ]


def test_run_comparison_stages_plot_paths_before_logging(tmp_path: Path) -> None:
    """Comparison artifacts should record uploaded figures paths, not temp absolutes."""
    external_plots = tmp_path / "external-plots"
    external_plots.mkdir()
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    convergence = external_plots / "convergence.png"
    condition_numbers = external_plots / "condition_numbers.png"
    convergence.write_text("convergence", encoding="utf-8")
    condition_numbers.write_text("condition", encoding="utf-8")
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    staged_plots_result = ComparisonResult(
        results={},
        summary="ok",
        solver_params=_solver_params(tmp_path),
        preconditioners=("none",),
        plot_paths=PlotPaths(
            convergence=convergence,
            condition_numbers=condition_numbers,
        ),
        recommendations=ComparisonRecommendations(),
    )
    logged_files: set[str] = set()
    comparison_json: dict[str, object] = {}
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    def _capture_work_root(tracking_uri: str, run_id: str, work_root: Path) -> None:
        nonlocal logged_files, comparison_json
        logged_files = {
            item.relative_to(work_root).as_posix()
            for item in work_root.rglob("*")
            if item.is_file()
        }
        comparison_json = json.loads((work_root / "comparison.json").read_text(encoding="utf-8"))

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=staged_plots_result),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS, side_effect=_capture_work_root),
    ):
        _configure_mock_mlflow(mock_mlflow)
        topology = _resolve_comparison_topology(experiments_config, settings)
        outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is True
    assert "figures/convergence.png" in logged_files
    assert "figures/condition_numbers.png" in logged_files
    plot_paths = comparison_json["plot_paths"]
    assert isinstance(plot_paths, dict)
    assert plot_paths["convergence"] == "figures/convergence.png"
    assert plot_paths["condition_numbers"] == "figures/condition_numbers.png"


def test_run_comparison_warns_and_continues_when_neural_resolution_fails(
    tmp_path: Path, stub_comparison_result: ComparisonResult
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY),
            NeuralPreconditionerConfig(
                name="missing-neural",
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref("MissingFFNN", "solutions"),
            ),
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=stub_comparison_result),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_preconditioner_models_with_warnings",
            return_value=MagicMock(
                specs=[StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)],
                warnings=(
                    "Skipping neural preconditioner 'missing-neural': Registered model 'MissingFFNN' not found",
                ),
            ),
        ),
    ):
        _configure_mock_mlflow(mock_mlflow)
        topology = _resolve_comparison_topology(experiments_config, settings)
        outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is True
    assert len(outcomes[0].warnings) == 1
    assert "Skipping neural preconditioner 'missing-neural'" in outcomes[0].warnings[0]


def test_run_comparison_fails_if_all_preconditioners_are_skipped(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            NeuralPreconditionerConfig(
                name="missing-neural",
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref("MissingFFNN", "solutions"),
            ),
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_preconditioner_models_with_warnings",
            return_value=MagicMock(
                specs=[],
                warnings=(
                    "Skipping neural preconditioner 'missing-neural': Registered model 'MissingFFNN' not found",
                ),
            ),
        ),
    ):
        _configure_mock_mlflow(mock_mlflow)
        topology = _resolve_comparison_topology(experiments_config, settings)
        outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is False
    assert "No runnable preconditioners remain" in (outcomes[0].error or "")


def test_run_comparison_ignores_unrelated_broken_experiments(
    tmp_path: Path, stub_comparison_result: ComparisonResult
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
    experiments_config = tmp_path / "experiments.toml"
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
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            NeuralPreconditionerConfig(
                name="registered-neural",
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref(
                    "NormScaledLinearFFNN",
                    "residuals-100",
                ),
            ),
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=stub_comparison_result),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_preconditioner_models_with_warnings",
            return_value=MagicMock(specs=cfg.preconditioners, warnings=()),
        ) as mock_resolve_specs,
    ):
        _configure_mock_mlflow(mock_mlflow)
        topology = _resolve_comparison_topology(experiments_config, settings)
        outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is True
    assert mock_resolve_specs.call_args.kwargs["assignment_contexts"] is None


def test_run_comparison_fails_before_tracking_when_matrix_input_is_missing(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    missing_matrix = tmp_path / "missing-matrix.npy"
    _, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=missing_matrix,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING) as mock_setup_tracking,
    ):
        topology = _resolve_comparison_topology(experiments_config, settings)
        outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is False
    assert f"Comparison input not found: {missing_matrix}" in (outcomes[0].error or "")
    mock_setup_tracking.assert_not_called()
    mock_mlflow.start_run.assert_not_called()


def test_run_comparison_missing_known_dataset_raises_file_not_found(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    missing_dataset = tmp_path / "solutions"
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, np.array([[1.0, 0.0]], dtype=np.float64))
    cfg = _mock_cfg(
        matrix_path=missing_dataset,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    topology = _resolve_comparison_topology(experiments_config, settings)
    outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is False
    assert str(missing_dataset) in (outcomes[0].error or "")


def test_run_comparison_batch_preserves_declared_order(tmp_path: Path) -> None:
    """Cache-hit outcomes come back in declared config order.

    Also proves no MLflow run is needed for this: run_comparison_batch never
    calls mlflow.start_run when _prepare_comparison_entry reports every entry
    as a cache hit.
    """
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config, with_comparisons=True)

    def _fake_prepare_entry(
        cfg: object,
        entry: ComparisonRegistryEntry,
        context: object,
        *,
        force: bool = False,
    ) -> ComparisonOutcome:
        return ComparisonOutcome(
            comparison_id=entry.id,
            comparison_display_name=entry.effective_display_name,
            success=True,
        )

    with (
        patch(
            "neuralls.composition.assignments.comparison_batch._prepare_comparison_entry",
            side_effect=_fake_prepare_entry,
        ) as mock_prepare,
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_config",
            return_value=_mock_cfg(),
        ),
        patch(_MLFLOW_MODULE) as mock_mlflow,
    ):
        outcomes = run_comparison_batch(experiments_config, ComparisonParams())

    assert [outcome.comparison_id for outcome in outcomes] == ["a", "b"]
    assert [call.args[1].id for call in mock_prepare.call_args_list] == ["a", "b"]
    mock_mlflow.start_run.assert_not_called()


def test_log_comparison_metrics_logs_scalar_metrics_per_preconditioner(
    tmp_path: Path,
) -> None:
    """_log_comparison_metrics must emit one metric set per preconditioner."""
    result = _typed_comparison_result(tmp_path / "conv.png")

    with patch(_COMPARISON_TRACKING_MLFLOW_MODULE) as mock_mlflow:
        log_comparison_result_metrics(result, child_run_tags={"none": {}})

    metric_calls = {call.args[0]: call.args[1] for call in mock_mlflow.log_metric.call_args_list}
    assert metric_calls["condition_number/none"] == 1.0
    assert metric_calls["iterations/none"] == 2
    assert metric_calls["final_residual/none"] == pytest.approx(1.0e-8)
    assert metric_calls["converged/none"] == 1
    mock_mlflow.log_param.assert_called_once_with("best_preconditioner", "none")


def test_log_linear_system_params_logs_matrix_and_rhs_shape(tmp_path: Path) -> None:
    """log_linear_system_params must log matrix/rhs shape, RHS kind, and raw condition number."""
    result = replace(
        _typed_comparison_result(tmp_path / "conv.png"),
        matrix_shape=(10, 10),
        rhs_shape=(10,),
        condition_number_raw=42.0,
        rhs_source_kind=ComparisonRhsSourceKind.SPARSE,
    )

    with patch(_COMPARISON_TRACKING_MLFLOW_MODULE) as mock_mlflow:
        log_linear_system_params(result)

    param_calls = {call.args[0]: call.args[1] for call in mock_mlflow.log_param.call_args_list}
    assert param_calls["matrix_rows"] == 10
    assert param_calls["matrix_cols"] == 10
    assert param_calls["rhs_dim"] == 10
    assert param_calls["rhs_source_kind"] == "sparse"
    mock_mlflow.log_metric.assert_called_once_with("condition_number_raw", 42.0)


def test_run_comparison_logs_scalar_metrics_to_mlflow(tmp_path: Path) -> None:
    """compare workflow must log per-preconditioner metrics to the parent run."""
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    plot_path = tmp_path / "conv.png"
    plot_path.write_text("x", encoding="utf-8")
    metrics_result = _typed_comparison_result(plot_path)
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=metrics_result),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
    ):
        _configure_mock_mlflow(mock_mlflow)
        topology = _resolve_comparison_topology(experiments_config, settings)
        outcomes = _run_comparison_from_config(cfg, entry, topology, experiments_config, settings)

    assert outcomes[0].success is True
    logged_metric_names = {call.args[0] for call in mock_mlflow.log_metric.call_args_list}
    assert "condition_number/none" in logged_metric_names
    assert "iterations/none" in logged_metric_names
    assert "final_residual/none" in logged_metric_names
    assert "converged/none" in logged_metric_names


def test_log_comparison_result_metrics_logs_child_scalar_metrics(tmp_path: Path) -> None:
    """Each child run must log scalar summary metrics for its preconditioner."""
    result = _typed_comparison_result(tmp_path / "conv.png")
    captured_metrics: list[tuple[str, float]] = []

    def _fake_start_run(*args: object, **kwargs: object) -> MagicMock:
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=MagicMock())
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with patch(_COMPARISON_TRACKING_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.side_effect = _fake_start_run
        mock_mlflow.log_metric.side_effect = lambda k, v, **kw: captured_metrics.append((k, v))
        log_comparison_result_metrics(result, child_run_tags={"none": {}})

    metric_map = {k: v for k, v in captured_metrics}
    assert metric_map["condition_number"] == 1.0
    assert metric_map["iterations"] == 2
    assert metric_map["final_residual"] == pytest.approx(1.0e-8)
    assert metric_map["converged"] == 1
