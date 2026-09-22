"""Unit tests for preconditioner factory function.

Tests factory creation, error handling, and edge cases independent of solver internals.
Follows project principles:
- Use fixtures for all test data
- Use tmp_path for temporary files (never tempfile)
- Type hints throughout
- Focus on our own logic, not torchalg internals
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import torch
from torch.testing import assert_close
from torchalg.preconditioners.base import PreconditionerContext
from torchalg.preconditioners.implementations import (
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
    NeuralPreconditioner,
    ScheduledPreconditioner,
)
from torchalg.preconditioners.implementations.amg import (
    AdaptiveSAPreconditioner,
    AggregationCoarsening,
    AMGPreconditioner,
    TargetDimensionCoarsening,
)
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter

from neuralls.composition.preconditioners.factory import (
    PreconditionerScheduleConfig,
    create_preconditioner,
    create_scheduled_preconditioner,
)
from neuralls.domain.inference_ports import InferencePredictorPort
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PowerNormWeightingConfig,
    PreconditionerType,
    SmootherPersistenceWeightingConfig,
    StandardPreconditionerConfig,
    TargetDimCoarseningConfig,
)

# ==============================================================================
# Mock Predictor and Adapter for Neural Preconditioner Testing
# ==============================================================================


class MockPredictor(ExtraInputPredictorPort):
    """Lightweight mock predictor for testing."""

    def __init__(self) -> None:
        """Initialize mock predictor."""
        self.cleaned_up = False
        self.apply_count = 0

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Return empty tuple — this mock needs no extra inputs."""
        return ()

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Mock apply: returns half of input."""
        self.apply_count += 1
        return residual * 0.5

    def cleanup(self) -> None:
        """Mark as cleaned up."""
        self.cleaned_up = True

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager with cleanup."""
        self.cleanup()


class MockAdapter(PredictorAdapter):
    """Lightweight mock adapter for testing."""

    def __init__(self) -> None:
        """Initialize mock adapter."""
        self.predictor = MockPredictor()

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Create mock predictor, raising if the checkpoint is missing."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return self.predictor


class MockInferencePredictor(InferencePredictorPort):
    """Lightweight mock batch-inference predictor for neural POD-2G testing."""

    def __init__(self, output_dim: int) -> None:
        """Initialize with the row width to return for each predicted snapshot."""
        self._output_dim = output_dim
        self.cleaned_up = False

    def predict_batch(self, feature_batch: dict[str, np.ndarray]) -> np.ndarray:
        """Mock prediction: one zero row per input sample."""
        n_samples = next(iter(feature_batch.values())).shape[0]
        return np.zeros((n_samples, self._output_dim))

    def cleanup(self) -> None:
        """Mark as cleaned up."""
        self.cleaned_up = True


def make_mock_inference_predictor_factory(
    output_dim: int,
) -> Callable[[Path, object], MockInferencePredictor]:
    """Build an `inference_predictor_factory` callable, raising on a missing checkpoint."""

    def factory(checkpoint_path: Path, settings: object) -> MockInferencePredictor:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return MockInferencePredictor(output_dim)

    return factory


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def well_conditioned_matrix() -> torch.Tensor:
    """Well-conditioned 4x4 SPD matrix for testing (condition number ~2)."""
    return torch.diag(torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch.float64))


@pytest.fixture
def dense_spd_matrix() -> torch.Tensor:
    """Dense 5x5 SPD tridiagonal matrix for ILU testing."""
    n = 5
    return (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )


@pytest.fixture
def residual_vector() -> torch.Tensor:
    """4D residual vector."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Create mock checkpoint file."""
    checkpoint = tmp_path / "mock_checkpoint.ckpt"
    checkpoint.write_text("mock checkpoint data")
    return checkpoint


@pytest.fixture
def pod_snapshot_dataset_dir(tmp_path: Path, well_conditioned_matrix: torch.Tensor) -> Path:
    """Minimal generated dataset directory supplying POD-2G snapshot solutions.

    Builds a manifest-backed dataset (matching the repo's canonical
    generation pipeline) with a handful of solution rows shaped to the 4x4
    `well_conditioned_matrix`, so `PODCoarseningConfig.dataset_dir` can be
    read through `load_dense_training_arrays` exactly like a real dataset.
    """
    from neuralls.domain.generation.payloads import GeneratedDatasetPayload
    from neuralls.platform.storage.datasets import DenseDatasetWriter, DenseZarrAccumulator

    matrix_np = well_conditioned_matrix.numpy()
    rng = np.random.default_rng(0)
    solutions = rng.standard_normal((4, matrix_np.shape[0]))
    rhs = solutions @ matrix_np.T

    dataset_dir = tmp_path / "pod-dataset"
    dataset_dir.mkdir()
    acc = DenseZarrAccumulator(dataset_dir / "matrix.zarr")
    acc.append_dense_matrix(matrix_np, repeats=1)
    zarr_path = acc.finalize()
    DenseDatasetWriter().write_dataset(
        dataset_dir,
        GeneratedDatasetPayload(
            rhs=rhs,
            solutions=solutions,
            matrix_artifact_path=zarr_path,
            matrix_size=(int(matrix_np.shape[0]), int(matrix_np.shape[1])),
            normalization_type="matrix",
            matrix_norm=float(np.linalg.norm(matrix_np, ord=2)),
            matrix_norm_type="spectral",
        ),
    )
    return dataset_dir


@pytest.fixture
def neural_pod_dataset_dir(tmp_path: Path, well_conditioned_matrix: torch.Tensor) -> Path:
    """Minimal manifest-backed dataset dir supplying one `params` array.

    Unlike `pod_snapshot_dataset_dir`, this dataset carries no meaningful
    `solutions` — the neural POD-2G branch never reads solutions off disk,
    it predicts the snapshot ensemble from `params` via a checkpoint. Uses
    plain `npy` artifacts (not `zarr`) since local zarr stores can hang in
    the sandboxed test environment. The array's on-disk artifact key is
    irrelevant to the model call — `NeuralPODCoarseningConfig.input_names`
    (not any dataset-side key) supplies the checkpoint's kwarg name(s).
    """
    from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
    from neuralls.platform.storage.manifest_io import (
        make_dataset_manifest,
        save_dataset_manifest,
    )

    n = well_conditioned_matrix.shape[0]
    samples = 4
    rng = np.random.default_rng(0)
    params = rng.standard_normal((samples, 2))

    np.save(tmp_path / "matrix.npy", well_conditioned_matrix.numpy())
    np.save(tmp_path / "rhs.npy", np.zeros((samples, n)))
    np.save(tmp_path / "solutions.npy", np.zeros((samples, n)))
    np.save(tmp_path / "params.npy", params)

    save_dataset_manifest(
        tmp_path,
        make_dataset_manifest(
            matrix=DatasetArtifact(path="matrix.npy", format="npy", dtype="float64", shape=(n, n)),
            rhs=DatasetArtifact(path="rhs.npy", format="npy", dtype="float64", shape=(samples, n)),
            solutions=DatasetArtifact(
                path="solutions.npy", format="npy", dtype="float64", shape=(samples, n)
            ),
            normalization=DatasetNormalization(
                type="none", matrix_norm=1.0, matrix_norm_type="fro", scale={}
            ),
            params=(
                DatasetArtifact(
                    path="params.npy",
                    format="npy",
                    dtype="float64",
                    shape=(samples, 2),
                ),
            ),
        ),
    )
    return tmp_path


# ==============================================================================
# Factory Tests - Identity
# ==============================================================================


def test_factory_creates_identity_preconditioner(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory creates Identity preconditioner for identity type."""
    config = StandardPreconditionerConfig(name="identity", type=PreconditionerType.IDENTITY)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, Identity)


def test_factory_creates_identity_for_none_type(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory creates Identity for 'none' type alias."""
    config = StandardPreconditionerConfig(name="none", type=PreconditionerType.NONE)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, Identity)


def test_identity_preconditioner_returns_copy(
    well_conditioned_matrix: torch.Tensor, residual_vector: torch.Tensor
) -> None:
    """Identity preconditioner returns an equal-valued clone of the input."""
    config = StandardPreconditionerConfig(name="identity", type=PreconditionerType.IDENTITY)

    precond = create_preconditioner(well_conditioned_matrix, config)
    result = precond.apply(residual_vector)

    assert_close(result, residual_vector)
    assert result is not residual_vector


# ==============================================================================
# Factory Tests - Jacobi
# ==============================================================================


def test_factory_creates_jacobi_preconditioner(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory creates JacobiPreconditioner for jacobi type."""
    config = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, JacobiPreconditioner)


def test_jacobi_preconditioner_applies_diagonal_scaling(
    well_conditioned_matrix: torch.Tensor, residual_vector: torch.Tensor
) -> None:
    """JacobiPreconditioner divides by the diagonal: z_i = r_i / A_ii."""
    config = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)
    result = precond.apply(residual_vector)

    expected = residual_vector / torch.diagonal(well_conditioned_matrix)
    assert_close(result, expected)


def test_jacobi_preserves_dtype(well_conditioned_matrix: torch.Tensor) -> None:
    """JacobiPreconditioner preserves float64 dtype."""
    config = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)
    residual = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    result = precond.apply(residual)

    assert result.dtype == torch.float64


# ==============================================================================
# Factory Tests - ILU
# ==============================================================================


def test_factory_creates_ilu_preconditioner(dense_spd_matrix: torch.Tensor) -> None:
    """Factory creates ILUPreconditioner for ilu type."""
    config = StandardPreconditionerConfig(name="ilu", type=PreconditionerType.ILU)

    precond = create_preconditioner(dense_spd_matrix, config)

    assert isinstance(precond, ILUPreconditioner)


def test_ilu_preconditioner_accepts_dense_matrix(dense_spd_matrix: torch.Tensor) -> None:
    """ILUPreconditioner applies without error on a dense tridiagonal matrix."""
    config = StandardPreconditionerConfig(name="ilu", type=PreconditionerType.ILU)

    precond = create_preconditioner(dense_spd_matrix, config)

    residual_5d = torch.ones(5, dtype=torch.float64)
    result = precond.apply(residual_5d)

    assert result.shape == (5,)
    assert result.dtype == torch.float64


def test_ilu_preserves_dtype(dense_spd_matrix: torch.Tensor) -> None:
    """ILUPreconditioner preserves float64 dtype."""
    config = StandardPreconditionerConfig(name="ilu", type=PreconditionerType.ILU)

    precond = create_preconditioner(dense_spd_matrix, config)
    residual = torch.ones(5, dtype=torch.float64)
    result = precond.apply(residual)

    assert result.dtype == torch.float64


# ==============================================================================
# Factory Tests - Neural
# ==============================================================================


def test_factory_creates_neural_preconditioner(
    well_conditioned_matrix: torch.Tensor, mock_checkpoint: Path
) -> None:
    """Factory creates NeuralPreconditioner for neural type."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)


def test_neural_preconditioner_with_custom_adapter(
    well_conditioned_matrix: torch.Tensor, mock_checkpoint: Path, residual_vector: torch.Tensor
) -> None:
    """NeuralPreconditioner delegates to the injected adapter's predictor (DIP)."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)
    result = precond.apply(residual_vector)

    expected = residual_vector * 0.5
    assert_close(result, expected)


def test_neural_preconditioner_checkpoint_not_found(
    well_conditioned_matrix: torch.Tensor,
    tmp_path: Path,
) -> None:
    """NeuralPreconditioner surfaces a missing checkpoint as FileNotFoundError."""
    mock_adapter = MockAdapter()
    missing_checkpoint = tmp_path / "missing" / "checkpoint.ckpt"

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=missing_checkpoint,
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)


def test_neural_preconditioner_cleanup_on_delete(
    well_conditioned_matrix: torch.Tensor, mock_checkpoint: Path, residual_vector: torch.Tensor
) -> None:
    """Neural preconditioner cleans up its predictor when cleanup() is called."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    predictor = mock_adapter.predictor
    assert not predictor.cleaned_up
    _ = precond.apply(residual_vector)

    precond.cleanup()

    assert predictor.cleaned_up


# ==============================================================================
# Factory Tests - AMG
# ==============================================================================


def test_factory_creates_amg_preconditioner_with_aggregation_coarsening(
    well_conditioned_matrix: torch.Tensor, residual_vector: torch.Tensor
) -> None:
    """Factory creates AMGPreconditioner for AMG type with aggregation coarsening."""
    config = AMGPreconditionerConfig(
        name="amg",
        coarsening=AggregationCoarseningConfig(theta=0.05, omega=0.5),
    )

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, AMGPreconditioner)
    assert isinstance(precond._coarsening, AggregationCoarsening)
    assert precond._coarsening._theta == 0.05
    assert precond._coarsening._omega == 0.5
    result = precond.apply(residual_vector)
    assert result.shape == residual_vector.shape


def test_factory_creates_amg_preconditioner_with_target_dim_coarsening(
    well_conditioned_matrix: torch.Tensor, residual_vector: torch.Tensor
) -> None:
    """Factory creates AMGPreconditioner for AMG type with target-dimension coarsening.

    Wiring only — TargetDimensionCoarsening's own search behavior is
    covered by torchalg's ``TestTargetDimensionCoarsening``; this just
    confirms the config's fields reach the constructed strategy unchanged.
    """
    config = AMGPreconditionerConfig(
        name="amg-target-dim",
        coarsening=TargetDimCoarseningConfig(
            target_coarse_dim=2, theta_min=0.1, theta_max=0.9, step=0.1, omega=0.5
        ),
    )

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, AMGPreconditioner)
    assert isinstance(precond._coarsening, TargetDimensionCoarsening)
    assert precond._coarsening._target_coarse_dim == 2
    assert precond._coarsening._theta_min == 0.1
    assert precond._coarsening._theta_max == 0.9
    assert precond._coarsening._step == 0.1
    assert precond._coarsening._omega == 0.5
    result = precond.apply(residual_vector)
    assert result.shape == residual_vector.shape


def test_factory_creates_amg_preconditioner_with_pod_coarsening(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
    pod_snapshot_dataset_dir: Path,
) -> None:
    """Factory creates AMGPreconditioner for AMG type with POD-2G coarsening.

    The snapshot ensemble is read from a generated dataset directory (via
    `load_dense_training_arrays`), not a raw glob — the gap this task closes.
    """
    coarsening = PODCoarseningConfig(dataset_dir=pod_snapshot_dataset_dir, rank=2)
    config = AMGPreconditionerConfig(name="pod2g", coarsening=coarsening)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, AMGPreconditioner)
    result = precond.apply(residual_vector)
    assert result.shape == residual_vector.shape


def test_factory_pod_coarsening_default_weighting_matches_unweighted_fit(
    well_conditioned_matrix: torch.Tensor,
    pod_snapshot_dataset_dir: Path,
) -> None:
    """No `weighting` configured must fit the exact same basis as before this feature existed."""
    from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

    from neuralls.platform.storage.dataset_readers import load_dense_training_arrays

    config = AMGPreconditionerConfig(
        name="pod2g", coarsening=PODCoarseningConfig(dataset_dir=pod_snapshot_dataset_dir, rank=2)
    )

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, AMGPreconditioner)
    assert isinstance(precond._coarsening, PODCoarseningStrategy)
    _, solutions = load_dense_training_arrays(pod_snapshot_dataset_dir)
    expected = PODCoarseningStrategy(rank=2)
    expected.fit(torch.as_tensor(solutions, dtype=well_conditioned_matrix.dtype))
    torch.testing.assert_close(precond._coarsening._basis, expected._basis)


def test_factory_pod_coarsening_power_norm_weighting_changes_the_basis(
    well_conditioned_matrix: torch.Tensor,
    pod_snapshot_dataset_dir: Path,
) -> None:
    """`weighting=power_norm` must produce a basis different from the unweighted fit.

    Confirms the config actually reaches the SVD, not just that it parses.
    """
    from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

    coarsening = PODCoarseningConfig(
        dataset_dir=pod_snapshot_dataset_dir,
        rank=2,
        weighting=PowerNormWeightingConfig(metric="l2", beta=1.0),
    )
    config = AMGPreconditionerConfig(name="pod2g", coarsening=coarsening)

    precond = create_preconditioner(well_conditioned_matrix, config)

    default_config = AMGPreconditionerConfig(
        name="pod2g", coarsening=PODCoarseningConfig(dataset_dir=pod_snapshot_dataset_dir, rank=2)
    )
    default_precond = create_preconditioner(well_conditioned_matrix, default_config)
    assert isinstance(precond, AMGPreconditioner)
    assert isinstance(default_precond, AMGPreconditioner)
    assert isinstance(precond._coarsening, PODCoarseningStrategy)
    assert isinstance(default_precond._coarsening, PODCoarseningStrategy)
    assert not torch.allclose(precond._coarsening._basis, default_precond._coarsening._basis)


def test_factory_pod_coarsening_smoother_persistence_weighting_runs_end_to_end(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
    pod_snapshot_dataset_dir: Path,
) -> None:
    """`weighting=smoother_persistence` (the scheme needing the system matrix) must work end-to-end."""
    coarsening = PODCoarseningConfig(
        dataset_dir=pod_snapshot_dataset_dir,
        rank=2,
        weighting=SmootherPersistenceWeightingConfig(omega=0.67, steps=3),
    )
    config = AMGPreconditionerConfig(name="pod2g", coarsening=coarsening)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, AMGPreconditioner)
    result = precond.apply(residual_vector)
    assert result.shape == residual_vector.shape


@pytest.fixture
def fitted_pod_checkpoint(tmp_path: Path, well_conditioned_matrix: torch.Tensor) -> Path:
    """A real dlkit-shaped checkpoint for an already-fitted `PODCoarseningStrategy`.

    Mirrors exactly the checkpoint dlkit's `OneShotFitExecutor` writes for a
    `run.type = "fit"` assignment (`state_dict` + `dlkit_metadata.model_settings`),
    so `build_model_from_checkpoint` reconstructs it the same way the factory's
    checkpoint-reconstruction branch does at real comparison time.
    """
    from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

    rng = np.random.default_rng(0)
    snapshots = torch.as_tensor(
        rng.standard_normal((4, well_conditioned_matrix.shape[0])), dtype=torch.float64
    )
    model = PODCoarseningStrategy(rank=2)
    model.fit(snapshots)

    checkpoint = {
        "state_dict": model.state_dict(),
        "dlkit_metadata": {
            "wrapper_type": "PODCoarseningStrategy",
            "model_settings": {
                "name": "PODCoarseningStrategy",
                "module_path": "torchalg.preconditioners.implementations.pod",
                "hyper_kwargs": {"rank": 2},
            },
            "entry_configs": [],
            "feature_names": [],
            "forward_arg_map": {},
            "predict_target_key": "",
            "model_family": "external",
            "input_shapes": None,
            "output_shapes": None,
        },
    }
    checkpoint_path = tmp_path / "pod-fit-checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def test_factory_creates_amg_preconditioner_from_fitted_pod_checkpoint(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
    fitted_pod_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """When `resolved_checkpoint_path` is set, the factory reconstructs the basis
    from the checkpoint instead of fitting inline from `dataset_dir`.

    `dataset_dir` is set to a non-dataset directory to prove it is never read
    once a checkpoint resolves — the fallback branch would raise trying to
    read `tmp_path` as a dataset.
    """
    coarsening = PODCoarseningConfig(
        dataset_dir=tmp_path,
        rank=2,
        resolved_checkpoint_path=fitted_pod_checkpoint,
    )
    config = AMGPreconditionerConfig(name="pod2g-fit", coarsening=coarsening)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, AMGPreconditioner)
    result = precond.apply(residual_vector)
    assert result.shape == residual_vector.shape


def test_factory_pod_checkpoint_reconstruction_rejects_non_pod_model(
    well_conditioned_matrix: torch.Tensor,
    tmp_path: Path,
) -> None:
    """A checkpoint reconstructing to a non-POD model raises a clear TypeError."""
    linear = torch.nn.Linear(2, 2, dtype=torch.float64)
    checkpoint = {
        "state_dict": linear.state_dict(),
        "dlkit_metadata": {
            "wrapper_type": "Linear",
            "model_settings": {
                "name": "Linear",
                "module_path": "torch.nn",
                "hyper_kwargs": {"in_features": 2, "out_features": 2},
            },
        },
    }
    checkpoint_path = tmp_path / "wrong-model-checkpoint.ckpt"
    torch.save(checkpoint, checkpoint_path)
    coarsening = PODCoarseningConfig(
        dataset_dir=tmp_path,
        rank=2,
        resolved_checkpoint_path=checkpoint_path,
    )
    config = AMGPreconditionerConfig(name="pod2g-fit", coarsening=coarsening)

    with pytest.raises(TypeError, match="expected PODCoarseningStrategy"):
        create_preconditioner(well_conditioned_matrix, config)


def test_factory_creates_amg_preconditioner_with_neural_pod_coarsening(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
    neural_pod_dataset_dir: Path,
    mock_checkpoint: Path,
) -> None:
    """Factory creates AMGPreconditioner for AMG type with neural POD-2G coarsening.

    The snapshot ensemble is predicted by a checkpoint (via the injected
    `inference_predictor_factory`), not read from a precomputed solutions
    array — the gap this task closes.
    """
    coarsening = NeuralPODCoarseningConfig(
        dataset_dir=neural_pod_dataset_dir,
        checkpoint_path=mock_checkpoint,
        input_names=("x",),
        rank=2,
    )
    config = AMGPreconditionerConfig(name="neural-pod2g", coarsening=coarsening)
    factory = make_mock_inference_predictor_factory(output_dim=well_conditioned_matrix.shape[0])

    precond = create_preconditioner(
        well_conditioned_matrix, config, inference_predictor_factory=factory
    )

    assert isinstance(precond, AMGPreconditioner)
    result = precond.apply(residual_vector)
    assert result.shape == residual_vector.shape


def test_factory_neural_pod_coarsening_requires_params_in_dataset(
    well_conditioned_matrix: torch.Tensor,
    pod_snapshot_dataset_dir: Path,
    mock_checkpoint: Path,
) -> None:
    """A dataset with no `params` array raises an actionable count-mismatch error.

    `pod_snapshot_dataset_dir` has rhs/solutions but no params (the common
    case — only one dataset in the whole repo populates params) — pointing
    neural_pod coarsening at it (with `input_names` declaring one expected
    input) must fail clearly, not with an opaque `RuntimeError: generator
    raised StopIteration` from an empty feature dict.
    """
    coarsening = NeuralPODCoarseningConfig(
        dataset_dir=pod_snapshot_dataset_dir,
        checkpoint_path=mock_checkpoint,
        input_names=("x",),
        rank=2,
    )
    config = AMGPreconditionerConfig(name="neural-pod2g", coarsening=coarsening)
    factory = make_mock_inference_predictor_factory(output_dim=well_conditioned_matrix.shape[0])

    with pytest.raises(ValueError, match="input_names has 1 name.*has 0 .params. array"):
        create_preconditioner(well_conditioned_matrix, config, inference_predictor_factory=factory)


def test_factory_neural_pod_coarsening_rejects_input_names_count_mismatch(
    well_conditioned_matrix: torch.Tensor,
    neural_pod_dataset_dir: Path,
    mock_checkpoint: Path,
) -> None:
    """Declaring more/fewer input_names than the dataset has params arrays is rejected.

    `neural_pod_dataset_dir` has exactly one `params` array; declaring two
    names must fail with a clear count mismatch, not silently zip-truncate
    or call the checkpoint with a wrong signature.
    """
    coarsening = NeuralPODCoarseningConfig(
        dataset_dir=neural_pod_dataset_dir,
        checkpoint_path=mock_checkpoint,
        input_names=("young_modulus", "poisson_ratio"),
        rank=2,
    )
    config = AMGPreconditionerConfig(name="neural-pod2g", coarsening=coarsening)
    factory = make_mock_inference_predictor_factory(output_dim=well_conditioned_matrix.shape[0])

    with pytest.raises(ValueError, match="input_names has 2 name.*has 1 .params. array"):
        create_preconditioner(well_conditioned_matrix, config, inference_predictor_factory=factory)


def test_factory_amg_requires_amg_config(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory requires AMGPreconditionerConfig for AMG type."""
    config = StandardPreconditionerConfig(name="amg", type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": PreconditionerType.AMG})

    with pytest.raises(TypeError, match="AMG type requires AMGPreconditionerConfig"):
        create_preconditioner(well_conditioned_matrix, config)


# ==============================================================================
# Factory Tests - Adaptive SA-AMG (alpha-SA)
# ==============================================================================


def test_factory_creates_adaptive_sa_preconditioner_with_fixed_theta(
    dense_spd_matrix: torch.Tensor,
) -> None:
    """Factory creates AdaptiveSAPreconditioner directly when no target dimension is set."""
    config = AdaptiveSAPreconditionerConfig(name="asa", max_coarse=1, theta=0.0)

    precond = create_preconditioner(dense_spd_matrix, config)

    assert isinstance(precond, AdaptiveSAPreconditioner)
    assert len(precond._result.matrices) == 2

    residual = torch.ones(5, dtype=torch.float64)
    result = precond.apply(residual)
    assert result.shape == (5,)


def test_factory_creates_adaptive_sa_preconditioner_with_explicit_num_candidates(
    dense_spd_matrix: torch.Tensor,
) -> None:
    """Factory forwards `num_candidates`/`theta` unchanged to `AdaptiveSAPreconditioner`.

    No target-dimension search (removed — an end-to-end run showed the
    search picking a worse theta than just naming one directly, see
    `docs/plan.md`); this just confirms config fields reach the built
    object unchanged.
    """
    config = AdaptiveSAPreconditionerConfig(name="asa", max_coarse=1, num_candidates=3, theta=0.1)

    precond = create_preconditioner(dense_spd_matrix, config)

    assert isinstance(precond, AdaptiveSAPreconditioner)
    assert precond._result.candidates.shape[1] == 3
    expected = AdaptiveSAPreconditioner(
        dense_spd_matrix, num_candidates=3, max_levels=2, max_coarse=1, theta=0.1
    )
    assert precond._result.matrices[-1].shape[0] == expected._result.matrices[-1].shape[0]


def test_factory_adaptive_sa_requires_adaptive_sa_config(
    well_conditioned_matrix: torch.Tensor,
) -> None:
    """Factory requires AdaptiveSAPreconditionerConfig for ADAPTIVE_SA_AMG type."""
    config = StandardPreconditionerConfig(name="asa", type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": PreconditionerType.ADAPTIVE_SA_AMG})

    with pytest.raises(
        TypeError, match="ADAPTIVE_SA_AMG type requires AdaptiveSAPreconditionerConfig"
    ):
        create_preconditioner(well_conditioned_matrix, config)


# ==============================================================================
# Factory Tests - Error Handling
# ==============================================================================


def test_factory_rejects_unsupported_type(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory raises ValueError for unsupported preconditioner type."""
    config = StandardPreconditionerConfig(name="invalid", type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": "invalid_type"})

    with pytest.raises(ValueError, match="Unsupported preconditioner type"):
        create_preconditioner(well_conditioned_matrix, config)


def test_factory_requires_neural_config_for_neural_type(
    well_conditioned_matrix: torch.Tensor,
) -> None:
    """Factory requires NeuralPreconditionerConfig for neural type."""
    config = StandardPreconditionerConfig(name="neural", type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": PreconditionerType.NEURAL})

    with pytest.raises(TypeError, match="Neural type requires NeuralPreconditionerConfig"):
        create_preconditioner(well_conditioned_matrix, config)


# ==============================================================================
# Scheduled Preconditioner Builder Tests
# ==============================================================================


def test_create_scheduled_preconditioner_no_scheduling() -> None:
    """Builder returns primary unchanged when no scheduling is needed."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=-1)
    result = create_scheduled_preconditioner(primary, schedule)

    assert result is primary


def test_create_scheduled_preconditioner_with_limit() -> None:
    """Builder wraps the preconditioner when limit_iters is specified."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=10)
    result = create_scheduled_preconditioner(primary, schedule)

    assert isinstance(result, ScheduledPreconditioner)
    assert result._limit_iters == 10
    assert result._start_iter == 0


def test_create_scheduled_preconditioner_with_start_iter() -> None:
    """Builder forwards delayed activation to ScheduledPreconditioner."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(start_iter=7, limit_iters=10)
    result = create_scheduled_preconditioner(primary, schedule)

    assert isinstance(result, ScheduledPreconditioner)
    assert result._start_iter == 7
    assert result._limit_iters == 10


def test_create_scheduled_preconditioner_wraps_unlimited_delayed_schedule() -> None:
    """Delayed unlimited schedules are still wrapped."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(start_iter=3, limit_iters=-1)
    result = create_scheduled_preconditioner(primary, schedule)

    assert isinstance(result, ScheduledPreconditioner)
    assert result._start_iter == 3
    assert result._limit_iters is None


def test_create_scheduled_preconditioner_delayed_schedule_uses_fallback_before_start(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Delayed factory schedules use the identity fallback before activation."""
    primary = JacobiPreconditioner(well_conditioned_matrix)
    schedule = PreconditionerScheduleConfig(start_iter=2, limit_iters=-1)
    result = create_scheduled_preconditioner(primary, schedule)

    early_ctx = PreconditionerContext(iteration=1, residual_norm=1.0, rhs_norm=1.0)
    active_ctx = PreconditionerContext(iteration=2, residual_norm=1.0, rhs_norm=1.0)

    assert_close(result.apply(residual_vector, early_ctx), residual_vector)
    assert_close(
        result.apply(residual_vector, active_ctx),
        torch.tensor([0.25, 2 / 3, 1.5, 2.0], dtype=torch.float64),
    )


def test_create_scheduled_preconditioner_default_fallback() -> None:
    """Identity is used as the default fallback."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=5)
    result = create_scheduled_preconditioner(primary, schedule)

    ctx = PreconditionerContext(iteration=10, residual_norm=1.0, rhs_norm=1.0)
    r = torch.tensor([1.0, 2.0], dtype=torch.float64)
    z = result.apply(r, ctx)

    assert_close(z, r)


def test_create_scheduled_preconditioner_with_identity_fallback() -> None:
    """Explicit Identity fallback type behaves the same as the default."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(
        limit_iters=5,
        fallback=PreconditionerType.IDENTITY,
    )
    result = create_scheduled_preconditioner(primary, schedule)

    ctx = PreconditionerContext(iteration=10, residual_norm=1.0, rhs_norm=1.0)
    r = torch.tensor([1.0, 2.0], dtype=torch.float64)
    z = result.apply(r, ctx)

    assert_close(z, r)
