"""Config loading smoke tests for training sections."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

from neuralls.composition.assignments.assembler import AssignmentIdentity, load_assignment

REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_relative_id(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


MINIMAL_FFNN_CONFIG = """
[run]
type = "train"
seed = 42
model = "model-profile.toml"
data = "model-profile.toml"

[training.trainer]
max_epochs = 1
[[training.trainer.callbacks]]
name = "EarlyStopping"
[[training.metrics]]
name = "RelativeVectorNormError"
"""

MINIMAL_LINEAR_CONFIG = """
[run]
type = "train"
seed = 42
model = "model-profile.toml"
data = "model-profile.toml"

[training.trainer]
max_epochs = 1
[[training.trainer.callbacks]]
name = "EarlyStopping"
"""

MINIMAL_GNN_CONFIG = """
[run]
type = "train"
seed = 42
model = "model-profile.toml"
data = "model-profile.toml"

[training.trainer]
max_epochs = 1
"""


@pytest.mark.parametrize(
    "config_content_template",
    [
        MINIMAL_FFNN_CONFIG,
        MINIMAL_LINEAR_CONFIG,
        MINIMAL_GNN_CONFIG,
    ],
)
def test_training_sections_round_trip(tmp_path: Path, config_content_template: str) -> None:
    """Ensure load_assignment preserves trainer callbacks/metrics from a temporary file."""
    # Create necessary directories that dlkit expects
    (tmp_path / "output").mkdir()

    # Inject the temporary path into the config content
    config_content = config_content_template.format(tmp_dir=tmp_path.as_posix())

    # Setup temporary config file
    config_path = tmp_path / "linear.toml"
    config_path.write_text(config_content)

    profile_content = {
        MINIMAL_FFNN_CONFIG: """
[model]
name = "ScaleEquivariantFFNN"

[data]
name = "FlexibleDataset"

[data.module]
name = "ArrayDataModule"
""",
        MINIMAL_LINEAR_CONFIG: """
[model]
name = "LinearModel"

[data]
name = "FlexibleDataset"

[data.module]
name = "ArrayDataModule"
""",
        MINIMAL_GNN_CONFIG: """
[model]
name = "GNNModel"
module_path = "dlkit.domain.nn.graph"

[data]
name = "GraphDataset"

[data.module]
name = "ArrayDataModule"
""",
    }[config_content_template]
    model_profile = tmp_path / "model-profile.toml"
    model_profile.write_text(profile_content)

    with config_path.open("rb") as fh:
        raw_config = tomllib.load(fh)

    raw_training = raw_config.get("training", {})
    raw_trainer = raw_training.get("trainer", {})
    raw_callbacks = tuple(raw_trainer.get("callbacks", ()))
    expected_callback_names = tuple(cb.get("name") for cb in raw_callbacks)

    raw_metrics = tuple(raw_training.get("metrics", ()))
    expected_metric_names = tuple(m.get("name") for m in raw_metrics)

    # Create dummy data config
    data_path = tmp_path / "data.toml"
    data_path.write_text('id="dummy_dataset"\n[source]\nmatrix_path="matrix.txt"\n')

    experiment = load_assignment(
        config_path,
        data_config_path=data_path,
        output_root=tmp_path / "output",
        identity=AssignmentIdentity(dataset_registry_id=data_path.stem),
    )
    settings = experiment.settings
    training = settings.training
    assert training is not None, "training section missing"
    assert settings.data is not None, "data section missing"

    actual_callback_names = tuple(cb.name for cb in training.trainer.callbacks)
    assert actual_callback_names[: len(expected_callback_names)] == expected_callback_names
    assert actual_callback_names[-1] == "RetainedCheckpointCopy"

    actual_metric_names = tuple(metric.name for metric in training.metrics)
    assert actual_metric_names == expected_metric_names

    dataset_name = tomllib.loads(profile_content).get("data", {}).get("name")
    assert settings.data.name == dataset_name


@pytest.mark.parametrize(
    "config_path",
    tuple((REPO_ROOT / "configs/profiles/training").rglob("*.toml")),
    ids=_repo_relative_id,
)
def test_training_profiles_reference_existing_dlkit_exports(config_path: Path) -> None:
    """Training profiles must only point at DLKit names that exist in the installed package."""
    with config_path.open("rb") as fh:
        raw_config = tomllib.load(fh)

    training = raw_config.get("training", {})
    loss = training.get("loss")

    if loss is not None:
        loss_module = importlib.import_module(loss["module_path"])
        assert hasattr(loss_module, loss["name"]), (
            f"{config_path} references missing loss {loss['module_path']}.{loss['name']}"
        )

    for metric in training.get("metrics", ()):
        metric_module = importlib.import_module(metric["module_path"])
        assert hasattr(metric_module, metric["name"]), (
            f"{config_path} references missing metric {metric['module_path']}.{metric['name']}"
        )


@pytest.mark.parametrize(
    "config_path",
    tuple((REPO_ROOT / "configs/jobs").rglob("*.toml")),
    ids=_repo_relative_id,
)
def test_checked_in_jobs_load_through_dlkit(config_path: Path) -> None:
    """Every checked-in job config must validate against the installed DLKit schema.

    Catches upstream DLKit field renames (e.g. training.loss_function -> training.loss)
    that synthetic test fixtures elsewhere don't exercise.
    """
    from dlkit.infrastructure.config.factories import load_job
    from dlkit.infrastructure.config.job_config import ConvergenceJobConfig, MultiRunJobConfig

    job = load_job(config_path)
    assert not isinstance(job, (ConvergenceJobConfig, MultiRunJobConfig)), (
        f"{config_path} is a sweep-level config, not a single job"
    )
    assert job.model is not None
    assert job.data is not None
