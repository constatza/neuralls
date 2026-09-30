"""Tests for `_generation_cost.py::resolve_generation_cost`.

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

from neuralls.composition.comparison._generation_cost import resolve_generation_cost
from neuralls.domain.solver.models.result import StageCost
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerType,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest as _make_manifest
from neuralls.platform.storage.manifest_io import save_dataset_manifest
from neuralls.shared.types import CostProvenance

_GENERATION_DURATION = 12.5


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
        name="amg", type=PreconditionerType.AMG, coarsening=AggregationCoarseningConfig()
    )
    assert resolve_generation_cost(cfg, settings=neuralls_settings) is None


def test_pod2g_with_stamped_manifest_reports_historical_cost(
    tmp_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """A POD-2G config's `dataset_dir` manifest with a real stamped duration
    resolves to a HISTORICAL StageCost with that exact wall time.
    """
    dataset_dir = tmp_path / "pod-dataset"
    _stamp_manifest(dataset_dir, generation_duration_seconds=_GENERATION_DURATION)
    cfg = AMGPreconditionerConfig(
        name="pod2g",
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=4),
    )

    result = resolve_generation_cost(cfg, settings=neuralls_settings)

    assert result == StageCost(
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
        name="pod2g",
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=4),
    )

    result = resolve_generation_cost(cfg, settings=neuralls_settings)

    assert result is not None
    assert result.provenance is CostProvenance.UNAVAILABLE


def test_pod2g_missing_dataset_dir_is_unavailable_not_an_exception(
    tmp_path: Path, neuralls_settings: NeurallsSettings
) -> None:
    """A `dataset_dir` pointing at a directory with no manifest at all (never
    generated) must resolve to UNAVAILABLE, never raise.
    """
    dataset_dir = tmp_path / "never-generated"
    cfg = AMGPreconditionerConfig(
        name="pod2g",
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=4),
    )

    result = resolve_generation_cost(cfg, settings=neuralls_settings)

    assert result is not None
    assert result.provenance is CostProvenance.UNAVAILABLE


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
        name="neural",
        type=PreconditionerType.NEURAL,
        data_config_path=data_config_path,
    )

    result = resolve_generation_cost(cfg, settings=neuralls_settings)

    assert result is not None
    assert result.provenance is CostProvenance.HISTORICAL
    assert result.wall_time_seconds == _GENERATION_DURATION


def test_neural_config_without_data_config_path_has_no_generation_cost(
    neuralls_settings: NeurallsSettings,
) -> None:
    """A Neural config with no `data_config_path` at all is not applicable."""
    cfg = NeuralPreconditionerConfig(name="neural", type=PreconditionerType.NEURAL)
    assert resolve_generation_cost(cfg, settings=neuralls_settings) is None
