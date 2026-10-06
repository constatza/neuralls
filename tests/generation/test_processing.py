"""Tests for typed generation processing entrypoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuralls.composition.generation.process_data import process_data_from_config
from neuralls.composition.generation.processing import _build_context, process_config
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.datasets import load_dense_training_arrays


def _write_solution_archive_config(tmp_path: Path, dataset_id: str) -> Path:
    """Create a minimal solution-archive config backed by local test files."""
    matrix_path = tmp_path / "matrix.txt"
    np.savetxt(matrix_path, np.eye(2))

    solutions_dir = tmp_path / "solutions"
    solutions_dir.mkdir()
    np.savetxt(solutions_dir / "solution_0.txt", np.array([1.0, 2.0]))

    config_path = tmp_path / f"{dataset_id}.toml"
    config_path.write_text(
        f"""
id = "{dataset_id}"

[source]
matrix_path = "{matrix_path.as_posix()}"
solutions_path = "{(solutions_dir / "solution_*.txt").as_posix()}"

[generation]
normalize = "matrix"

[[generation.strategy]]
name = "solution_archive"
samples = -1
solutions_glob = "{(solutions_dir / "solution_*.txt").as_posix()}"

[output]
"""
    )
    return config_path


def _write_enumerated_generation_config(tmp_path: Path, dataset_id: str) -> Path:
    """Create a generation config that relies on enumerate_by for globbed matrices."""
    matrices_dir = tmp_path / "matrices"
    matrices_dir.mkdir()
    for idx, stem in enumerate(
        (
            "E1_10554158_E2_20662154_E3_19907116_E4_20715669_subdomain_1_Kaa",
            "E1_10846921_E2_25611823_E3_31377296_E4_28205425_subdomain_1_Kaa",
        )
    ):
        np.savetxt(matrices_dir / f"{stem}.txt", np.eye(2) * float(idx + 2))

    config_path = tmp_path / f"{dataset_id}.toml"
    config_path.write_text(
        f"""
id = "{dataset_id}"

[source]
matrix_path = "{(matrices_dir / "*_subdomain_1_Kaa.txt").as_posix()}"
enumerate_by = "name"

[generation]
normalize = "none"

[[generation.strategy]]
name = "neutral_ones"
samples = 1

[output]
"""
    )
    return config_path


def _write_uniform_generation_config(tmp_path: Path, dataset_id: str) -> Path:
    """Create a 3-matrix config whose 5 samples split uniformly: 2, 2 and 1 per matrix."""
    matrices_dir = tmp_path / "replacement_matrices"
    matrices_dir.mkdir()
    for idx, stem in enumerate(("matrix_alpha", "matrix_beta", "matrix_gamma")):
        np.savetxt(matrices_dir / f"{stem}.txt", np.eye(2) * float(idx + 2))

    config_path = tmp_path / f"{dataset_id}.toml"
    config_path.write_text(
        f"""
id = "{dataset_id}"

[source]
matrix_path = "{(matrices_dir / "matrix_*.txt").as_posix()}"
enumerate_by = "name"

[generation]
normalize = "none"

[[generation.strategy]]
name = "random"
samples = 5

[output]
"""
    )
    return config_path


def test_process_config_accepts_dataconfig_file(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """`process_config` consumes typed DataConfigFile objects."""
    config_path = _write_solution_archive_config(tmp_path, "typed-dataset")
    config = load_data_config(config_path, neuralls_settings).model_copy(
        update={
            "output": load_data_config(config_path, neuralls_settings).output.model_copy(
                update={"data_dir": neuralls_settings.processed_dir}
            )
        }
    )
    matrix = np.loadtxt(config.source.matrix_path)
    output_dir = process_config(config, matrix)
    assert output_dir == neuralls_settings.processed_dir / "typed-dataset"


def test_data_generation_context_from_typed_config(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Context assembly reads typed fields instead of raw mappings."""
    config_path = _write_solution_archive_config(tmp_path, "context-dataset")
    config = load_data_config(config_path, neuralls_settings).model_copy(
        update={
            "output": load_data_config(config_path, neuralls_settings).output.model_copy(
                update={"data_dir": neuralls_settings.processed_dir}
            )
        }
    )
    context, _ = _build_context(config=config)
    assert context.matrix_path == config.source.matrix_path
    assert context.solutions_path == config.source.solutions_path
    assert context.enumerate_by == config.source.enumerate_by
    assert context.replacement is config.generation.replacement
    assert context.dataset_dir == neuralls_settings.processed_dir / "context-dataset"
    assert context.seed == config.generation.seed
    assert context.shuffle is config.generation.shuffle


def test_process_data_from_config_end_to_end(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """The loader-backed generation entrypoint builds a dataset under processed_dir."""
    config_path = _write_solution_archive_config(tmp_path, "end-to-end-dataset")
    output_dir = process_data_from_config(config_path, neuralls_settings)
    assert output_dir == neuralls_settings.processed_dir / "end-to-end-dataset"
    assert output_dir.exists()


def test_build_context_preserves_enumerate_by_from_config(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    config_path = _write_enumerated_generation_config(tmp_path, "enumerated-context")
    config = load_data_config(config_path, neuralls_settings).model_copy(
        update={
            "output": load_data_config(config_path, neuralls_settings).output.model_copy(
                update={"data_dir": neuralls_settings.processed_dir}
            )
        }
    )

    context, _ = _build_context(config=config)

    assert context.enumerate_by is not None
    assert context.enumerate_by.value == "name"


def test_process_data_from_config_honors_enumerate_by(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    config_path = _write_enumerated_generation_config(tmp_path, "enumerated-end-to-end")

    output_dir = process_data_from_config(config_path, neuralls_settings)

    assert output_dir == neuralls_settings.processed_dir / "enumerated-end-to-end"
    rhs, _ = load_dense_training_arrays(output_dir)
    assert rhs.shape[0] == 1


def test_build_context_replacement_is_false_for_uniform_config(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    config_path = _write_uniform_generation_config(tmp_path, "uniform-context")
    config = load_data_config(config_path, neuralls_settings).model_copy(
        update={
            "output": load_data_config(config_path, neuralls_settings).output.model_copy(
                update={"data_dir": neuralls_settings.processed_dir}
            )
        }
    )

    context, _ = _build_context(config=config)

    assert context.replacement is False


def test_process_data_from_config_splits_samples_uniformly(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    config_path = _write_uniform_generation_config(tmp_path, "uniform-end-to-end")

    output_dir = process_data_from_config(config_path, neuralls_settings)

    rhs, solutions = load_dense_training_arrays(output_dir)
    assert output_dir == neuralls_settings.processed_dir / "uniform-end-to-end"
    assert rhs.shape == (5, 2)
    assert solutions.shape == (5, 2)
