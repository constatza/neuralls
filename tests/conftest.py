"""Test fixtures for graph-cg tests."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from loguru import logger

from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.settings import NeurallsSettings

REPO_ROOT = Path(__file__).resolve().parent.parent


def _patch_default_paths_for_loaded_modules(
    path_values: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch already-imported modules that cache default path constants."""
    str_values = {
        "DEFAULT_PROCESSED_DIR": str(path_values["DEFAULT_PROCESSED_DATA_DIR"]),
        "DEFAULT_RESULTS_DIR": str(path_values["DEFAULT_OUTPUT_DIR"]),
        "DEFAULT_FIGURES_DIR_STR": str(path_values["DEFAULT_FIGURES_DIR"]),
    }
    all_values: dict[str, object] = {**path_values, **str_values}
    all_values["DEFAULT_CHECKPOINTS_DIR"] = path_values["DEFAULT_OUTPUT_DIR"] / "checkpoints"
    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        for name, value in all_values.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value, raising=False)


def _patch_dlkit_environment(
    runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch DLKit's imported environment singleton to the per-test root."""
    from dlkit.infrastructure.config.environment import env as dlkit_env

    monkeypatch.setattr(dlkit_env, "internal_dir", str(runtime_root / ".dlkit"), raising=False)


@pytest.fixture
def runtime_root(tmp_path: Path) -> Path:
    """Per-test runtime root for all generated artifacts."""
    return (tmp_path / "runtime").resolve()


@pytest.fixture
def output_root(runtime_root: Path) -> Path:
    """Per-test output root."""
    return runtime_root / "output"


@pytest.fixture
def processed_root(runtime_root: Path) -> Path:
    """Per-test processed data root."""
    return runtime_root / "processed"


@pytest.fixture
def figures_root(runtime_root: Path) -> Path:
    """Per-test figures root."""
    return runtime_root / "figures"


@pytest.fixture
def mlflow_tracking_dir(runtime_root: Path) -> Path:
    """Per-test MLflow tracking root."""
    return runtime_root / "mlruns"


@pytest.fixture
def mlflow_artifact_dir(runtime_root: Path) -> Path:
    """Per-test MLflow artifact root."""
    return runtime_root / "mlartifacts"


@pytest.fixture(autouse=True)
def isolate_default_paths_with_tmp_path(
    runtime_root: Path,
    output_root: Path,
    processed_root: Path,
    figures_root: Path,
    mlflow_tracking_dir: Path,
    mlflow_artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force all default artifact paths into pytest temp storage."""
    case_config_path = runtime_root / "case.toml"
    case_config_path.parent.mkdir(parents=True, exist_ok=True)
    case_config_path.write_text("", encoding="utf-8")
    env_map = {
        "NEURALLS_PROCESSED_DIR": str(processed_root),
        "NEURALLS_OUTPUT_DIR": str(output_root),
        "NEURALLS_CASE_CONFIG": str(case_config_path),
        # Force DLKit test artifacts away from repository-level tests/artifacts/.
        "DLKIT_TEST_MODE": "1",
        "DLKIT_TEST_ARTIFACT_ROOT": str(runtime_root / "dlkit_artifacts"),
        # Keep DLKit internals local to the test temp root.
        "DLKIT_ROOT_DIR": str(runtime_root),
        "DLKIT_INTERNAL_DIR": str(runtime_root / ".dlkit"),
    }
    (runtime_root / "raw").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(runtime_root)
    for key, value in env_map.items():
        monkeypatch.setenv(key, value)

    from neuralls.shared import constants as neuralls_constants

    path_values = {
        "DEFAULT_OUTPUT_DIR": Path(env_map["NEURALLS_OUTPUT_DIR"]).resolve(),
        "DEFAULT_PROCESSED_DATA_DIR": Path(env_map["NEURALLS_PROCESSED_DIR"]).resolve(),
        "DEFAULT_FIGURES_DIR": figures_root.resolve(),
        "DEFAULT_MLRUNS_DIR": mlflow_tracking_dir.resolve(),
        "DEFAULT_MLARTIFACTS_DIR": mlflow_artifact_dir.resolve(),
    }
    for name, value in path_values.items():
        monkeypatch.setattr(neuralls_constants, name, value, raising=False)

    _patch_default_paths_for_loaded_modules(path_values, monkeypatch)
    _patch_dlkit_environment(runtime_root, monkeypatch)


@pytest.fixture
def neuralls_settings(
    runtime_root: Path, processed_root: Path, output_root: Path
) -> NeurallsSettings:
    """Resolved settings fixture backed by per-test temporary roots."""
    raw = runtime_root / "raw"
    for directory in (raw, processed_root, output_root):
        directory.mkdir(parents=True, exist_ok=True)
    return NeurallsSettings(
        _env_file=[],
        processed_dir=processed_root,
        output_dir=output_root,
    )


@pytest.fixture
def config_context(neuralls_settings: NeurallsSettings, tmp_path: Path) -> ConfigContext:
    """Config context fixture anchored to a temporary config path."""
    config_path = tmp_path / "config.toml"
    config_path.touch()
    return ConfigContext(config_path=config_path, settings=neuralls_settings)


@pytest.fixture
def minimal_data_config_toml(
    tmp_path: Path,
    runtime_root: Path,
) -> Path:
    """Minimal valid dataset config for loader and processing tests."""
    config_path = tmp_path / "test.toml"
    processed = runtime_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    matrix_path = processed / "matrix.mtx"
    matrix_path.touch()
    config_path.write_text(
        """
id = "test-dataset"

[source]
matrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"

[generation]

[output]
""".strip()
    )
    return config_path


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure loguru to be safe against closed streams during cleanup."""

    def safe_sink(message):
        try:
            sys.stderr.write(message)
        except ValueError:
            pass  # Ignore "I/O operation on closed file"

    logger.remove()
    logger.add(safe_sink)


@pytest.fixture
def small_spd_matrix() -> np.ndarray:
    """Small 2x2 symmetric positive definite test matrix.

    Returns:
        2x2 SPD matrix with known properties
    """
    return np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


@pytest.fixture
def small_rhs() -> np.ndarray:
    """Small 2D RHS vector for testing.

    Returns:
        2D RHS vector
    """
    return np.array([1.0, 0.0], dtype=np.float64)


@pytest.fixture
def archive_solutions() -> np.ndarray:
    """Pre-computed archive solutions for testing.

    Returns:
        Array of 3 solution vectors (2D each)
    """
    return np.array(
        [[0.5, 0.3], [0.2, 0.8], [0.1, 0.4]],
        dtype=np.float64,
    )


@pytest.fixture
def archive_rhs(small_spd_matrix: np.ndarray, archive_solutions: np.ndarray) -> np.ndarray:
    """Pre-computed RHS vectors from archive solutions.

    Args:
        small_spd_matrix: Test matrix A
        archive_solutions: Archive solution vectors

    Returns:
        Array of RHS vectors b = A @ x
    """
    return np.array(
        [small_spd_matrix @ x for x in archive_solutions],
        dtype=np.float64,
    )


@pytest.fixture
def test_seed() -> int:
    """Deterministic seed for reproducible tests.

    Returns:
        Random seed value
    """
    return 42


@pytest.fixture
def sample_dataset_npz(tmp_path, small_spd_matrix, archive_solutions, archive_rhs):
    """Create a sample dataset using split-array sparse storage.

    Args:
        tmp_path: Pytest temporary directory fixture
        small_spd_matrix: Test matrix
        archive_solutions: Test solutions
        archive_rhs: Test RHS vectors

    Returns:
        Path to dataset directory containing manifest + arrays
    """
    dataset_dir = tmp_path / "test-dataset"
    dataset_dir.mkdir()
    from neuralls.domain.generation.payloads import GeneratedDatasetPayload
    from neuralls.platform.storage.datasets import DenseDatasetWriter, DenseZarrAccumulator

    acc = DenseZarrAccumulator(dataset_dir / "matrix.zarr")
    acc.append_dense_matrix(small_spd_matrix, repeats=1)
    zarr_path = acc.finalize()
    DenseDatasetWriter().write_dataset(
        dataset_dir,
        GeneratedDatasetPayload(
            rhs=archive_rhs,
            solutions=archive_solutions,
            matrix_artifact_path=zarr_path,
            matrix_size=(int(small_spd_matrix.shape[0]), int(small_spd_matrix.shape[1])),
            normalization_type="matrix",
            matrix_norm=float(np.linalg.norm(small_spd_matrix, ord=2)),
            matrix_norm_type="spectral",
        ),
    )

    return dataset_dir


@pytest.fixture
def sample_dataset_with_raw(tmp_path, small_spd_matrix, archive_solutions, archive_rhs):
    """Create a sample dataset with normalized artifacts and raw arrays.

    Args:
        tmp_path: Pytest temporary directory fixture
        small_spd_matrix: Test matrix
        archive_solutions: Test solutions
        archive_rhs: Test RHS vectors

    Returns:
        Path to dataset directory containing normalized artifacts and raw/ arrays
    """
    dataset_dir = tmp_path / "test-dataset-with-raw"
    dataset_dir.mkdir()
    from neuralls.domain.generation.payloads import GeneratedDatasetPayload
    from neuralls.platform.storage.datasets import DenseDatasetWriter, DenseZarrAccumulator

    acc = DenseZarrAccumulator(dataset_dir / "matrix.zarr")
    acc.append_dense_matrix(small_spd_matrix, repeats=1)
    zarr_path = acc.finalize()
    DenseDatasetWriter().write_dataset(
        dataset_dir,
        GeneratedDatasetPayload(
            rhs=archive_rhs,
            solutions=archive_solutions,
            matrix_artifact_path=zarr_path,
            matrix_size=(int(small_spd_matrix.shape[0]), int(small_spd_matrix.shape[1])),
            normalization_type="matrix",
            matrix_norm=float(np.linalg.norm(small_spd_matrix, ord=2)),
            matrix_norm_type="spectral",
        ),
    )

    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir()
    np.save(raw_dir / "matrix.npy", small_spd_matrix)
    np.save(raw_dir / "rhs.npy", archive_rhs)
    np.save(raw_dir / "solutions.npy", archive_solutions)

    return dataset_dir


@pytest.fixture
def warning_messages() -> Iterator[list[str]]:
    """Collect loguru WARNING-level messages emitted during a test."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    yield messages
    logger.remove(sink_id)
