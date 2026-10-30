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
from pydantic import ValidationError
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
    BootstrapAMGPreconditioner,
    TargetDimensionCoarsening,
)
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter

from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.composition.preconditioners.schedule import (
    PreconditionerScheduleConfig,
    create_scheduled_preconditioner,
)
from neuralls.domain.inference_ports import InferencePredictorPort
from neuralls.platform.config.models.coarsening import (
    AggregationCoarseningConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    TargetDimCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
    parse_preconditioner_config,
)
from neuralls.platform.dlkit.predictor_adapter import OwningDLKitNeuralPreconditioner
from tests.fakes.fit_data import FakeFitData

pytestmark = pytest.mark.deferred_io("config-loading")

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
) -> Callable[[Path], MockInferencePredictor]:
    """Build an `inference_predictor_factory` callable, raising on a missing checkpoint."""

    def factory(checkpoint_path: Path) -> MockInferencePredictor:
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
def dense_spd_matrix_20() -> torch.Tensor:
    """Dense 20x20 SPD tridiagonal matrix, large enough for Bootstrap AMG to coarsen."""
    n = 20
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
    checkpoint.write_text("mock checkpoint data", encoding="utf-8")
    return checkpoint


FIT_DIR = Path("pod-fit")


@pytest.fixture
def pod_fit_arrays(well_conditioned_matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """In-memory (rhs, solutions) fit arrays for the 4x4 system, as the service stages them."""
    n = well_conditioned_matrix.shape[0]
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, n)), rng.standard_normal((4, n))


@pytest.fixture
def parameter_arrays() -> tuple[np.ndarray, ...]:
    """One in-memory `params` array of four samples with two columns."""
    return (np.random.default_rng(0).standard_normal((4, 2)),)


def test_factory_creates_identity_preconditioner(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory creates Identity preconditioner for identity type."""
    config = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, Identity)


@pytest.mark.parametrize("invalid_type", ["none", "null"])
def test_preconditioner_config_rejects_removed_identity_aliases(invalid_type: str) -> None:
    """Removed no-op aliases fail at the typed Pydantic configuration boundary."""
    with pytest.raises(ValidationError):
        parse_preconditioner_config({"name": "identity", "type": invalid_type})


# ==============================================================================
# Factory Tests - Jacobi
# ==============================================================================


def test_factory_creates_jacobi_preconditioner(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory creates JacobiPreconditioner for jacobi type."""
    config = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, JacobiPreconditioner)


# ==============================================================================
# Factory Tests - ILU
# ==============================================================================


def test_factory_creates_ilu_preconditioner(dense_spd_matrix: torch.Tensor) -> None:
    """Factory creates ILUPreconditioner for ilu type."""
    config = StandardPreconditionerConfig(type=PreconditionerType.ILU)

    precond = create_preconditioner(dense_spd_matrix, config)

    assert isinstance(precond, ILUPreconditioner)


# ==============================================================================
# Factory Tests - Neural
# ==============================================================================


def test_factory_creates_neural_preconditioner(
    well_conditioned_matrix: torch.Tensor, mock_checkpoint: Path
) -> None:
    """Factory creates NeuralPreconditioner for neural type."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
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
        type=PreconditionerType.NEURAL,
        checkpoint_path=missing_checkpoint,
    )

    with pytest.raises(FileNotFoundError):
        create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)


def test_neural_preconditioner_cleanup_on_delete(
    well_conditioned_matrix: torch.Tensor, mock_checkpoint: Path, residual_vector: torch.Tensor
) -> None:
    """Neural preconditioner cleans up its predictor when cleanup() is called."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, OwningDLKitNeuralPreconditioner)
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
    pod_fit_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """Factory creates AMGPreconditioner for AMG type with POD-2G coarsening.

    The snapshot ensemble arrives staged, as `PreconditionerService.preload` supplies it.
    """
    config = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR, rank=2))

    precond = create_preconditioner(well_conditioned_matrix, config, staged_fit_data=pod_fit_arrays)

    assert isinstance(precond, AMGPreconditioner)


def test_inline_pod_coarsening_refuses_to_read_its_dataset_during_the_build(
    well_conditioned_matrix: torch.Tensor,
) -> None:
    """Unstaged fit data is an error, never a hidden read inside the timed build."""
    config = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR, rank=2))

    with pytest.raises(ValueError, match="not staged"):
        create_preconditioner(well_conditioned_matrix, config)


def test_pod_fit_arrays_are_staged_before_factory_build(
    well_conditioned_matrix: torch.Tensor,
    pod_fit_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """The service reads the fit arrays once in `preload`; the build itself reads nothing."""
    from neuralls.composition.comparison._preconditioner_setup import PreconditionerService

    config = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=FIT_DIR, rank=2))
    rhs, solutions = pod_fit_arrays
    fit_data = FakeFitData(rhs=rhs, solutions=solutions)
    service = PreconditionerService(fit_data)
    service.preload(well_conditioned_matrix, config)

    precond = service.create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, AMGPreconditioner)
    assert fit_data.dense_reads == [FIT_DIR]


def test_factory_creates_amg_preconditioner_with_neural_pod_coarsening(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
    parameter_arrays: tuple[np.ndarray, ...],
    mock_checkpoint: Path,
) -> None:
    """Factory creates AMGPreconditioner for AMG type with neural POD-2G coarsening.

    The snapshot ensemble is predicted by a checkpoint (via the injected
    `inference_predictor_factory`), not read from a precomputed solutions
    array — the gap this task closes.
    """
    coarsening = NeuralPODCoarseningConfig(
        dataset_dir=FIT_DIR,
        checkpoint_path=mock_checkpoint,
        input_names=("x",),
        rank=2,
    )
    config = AMGPreconditionerConfig(coarsening=coarsening)
    factory = make_mock_inference_predictor_factory(output_dim=well_conditioned_matrix.shape[0])

    precond = create_preconditioner(
        well_conditioned_matrix,
        config,
        inference_predictor_factory=factory,
        staged_fit_data=parameter_arrays,
    )

    assert isinstance(precond, AMGPreconditioner)
    result = precond.apply(residual_vector)
    assert result.shape == residual_vector.shape


def test_factory_neural_pod_coarsening_requires_params_in_dataset(
    well_conditioned_matrix: torch.Tensor,
    mock_checkpoint: Path,
) -> None:
    """A dataset with no `params` array raises a count-mismatch error.

    A dataset with rhs/solutions but no params (the common case) staged for neural_pod
    coarsening (with `input_names` declaring one expected input) must raise `ValueError`,
    rather than leaking `StopIteration` from an empty feature dictionary.
    """
    coarsening = NeuralPODCoarseningConfig(
        dataset_dir=FIT_DIR,
        checkpoint_path=mock_checkpoint,
        input_names=("x",),
        rank=2,
    )
    config = AMGPreconditionerConfig(coarsening=coarsening)
    factory = make_mock_inference_predictor_factory(output_dim=well_conditioned_matrix.shape[0])

    with pytest.raises(ValueError):
        create_preconditioner(
            well_conditioned_matrix,
            config,
            inference_predictor_factory=factory,
            staged_fit_data=(),
        )


def test_factory_neural_pod_coarsening_rejects_input_names_count_mismatch(
    well_conditioned_matrix: torch.Tensor,
    parameter_arrays: tuple[np.ndarray, ...],
    mock_checkpoint: Path,
) -> None:
    """Declaring more/fewer input_names than the dataset has params arrays is rejected.

    The staged dataset has exactly one `params` array; declaring two
    names must fail with a clear count mismatch, not silently zip-truncate
    or call the checkpoint with a wrong signature.
    """
    coarsening = NeuralPODCoarseningConfig(
        dataset_dir=FIT_DIR,
        checkpoint_path=mock_checkpoint,
        input_names=("young_modulus", "poisson_ratio"),
        rank=2,
    )
    config = AMGPreconditionerConfig(coarsening=coarsening)
    factory = make_mock_inference_predictor_factory(output_dim=well_conditioned_matrix.shape[0])

    with pytest.raises(ValueError):
        create_preconditioner(
            well_conditioned_matrix,
            config,
            inference_predictor_factory=factory,
            staged_fit_data=parameter_arrays,
        )


def test_factory_amg_requires_amg_config(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory requires AMGPreconditionerConfig for AMG type."""
    config = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": PreconditionerType.AMG})

    with pytest.raises(TypeError):
        create_preconditioner(well_conditioned_matrix, config)


# ==============================================================================
# Factory Tests - Adaptive SA-AMG (alpha-SA)
# ==============================================================================


def test_factory_creates_adaptive_sa_preconditioner_with_fixed_theta(
    dense_spd_matrix: torch.Tensor,
) -> None:
    """Factory creates AdaptiveSAPreconditioner directly when no target dimension is set."""
    config = AdaptiveSAPreconditionerConfig(max_coarse=1, theta=0.0)

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
    config = AdaptiveSAPreconditionerConfig(max_coarse=1, num_candidates=3, theta=0.1)

    precond = create_preconditioner(dense_spd_matrix, config)

    assert isinstance(precond, AdaptiveSAPreconditioner)
    assert precond._result.candidates.shape[1] == 3
    expected = AdaptiveSAPreconditioner(
        num_candidates=3, max_levels=2, max_coarse=1, theta=0.1
    ).setup(dense_spd_matrix)
    assert precond._result.matrices[-1].shape[0] == expected._result.matrices[-1].shape[0]


def test_factory_adaptive_sa_requires_adaptive_sa_config(
    well_conditioned_matrix: torch.Tensor,
) -> None:
    """Factory requires AdaptiveSAPreconditionerConfig for ADAPTIVE_SA_AMG type."""
    config = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": PreconditionerType.ADAPTIVE_SA_AMG})

    with pytest.raises(TypeError):
        create_preconditioner(well_conditioned_matrix, config)


# ==============================================================================
# Factory Tests - Bootstrap AMG (BAMG)
# ==============================================================================


def test_factory_creates_bootstrap_amg_preconditioner(
    dense_spd_matrix_20: torch.Tensor,
) -> None:
    """Factory creates BootstrapAMGPreconditioner and forwards config fields unchanged.

    `dense_spd_matrix` (5x5, used elsewhere in this file) is too small for
    BAMG's compatible-relaxation coarsening to produce more than one level
    (matching the same finding behind `tridiag_spd_matrix_20` in
    `test_preconditioner_labels.py`) — hence the larger dedicated fixture.
    """
    config = BootstrapAMGPreconditionerConfig(max_coarse=5, seed=0)

    precond = create_preconditioner(dense_spd_matrix_20, config)

    assert isinstance(precond, BootstrapAMGPreconditioner)
    expected = BootstrapAMGPreconditioner(max_coarse=5, seed=0).setup(dense_spd_matrix_20)
    assert precond._result.matrices[-1].shape[0] == expected._result.matrices[-1].shape[0]
    # n_levels=10 default is a ceiling on torchalg's coarsening loop, not an
    # exact count — real depth here is governed by max_coarse and settles
    # well under the ceiling (see BootstrapAMGPreconditionerConfig's docstring).
    assert len(precond._result.matrices) == len(expected._result.matrices) < config.n_levels

    residual = torch.ones(20, dtype=torch.float64)
    result = precond.apply(residual)
    assert result.shape == (20,)


def test_factory_bootstrap_amg_requires_bootstrap_amg_config(
    well_conditioned_matrix: torch.Tensor,
) -> None:
    """Factory requires BootstrapAMGPreconditionerConfig for BOOTSTRAP_AMG type."""
    config = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": PreconditionerType.BOOTSTRAP_AMG})

    with pytest.raises(TypeError):
        create_preconditioner(well_conditioned_matrix, config)


# ==============================================================================
# Factory Tests - Error Handling
# ==============================================================================


def test_factory_rejects_unsupported_type(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory raises ValueError for unsupported preconditioner type."""
    config = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": "invalid_type"})

    with pytest.raises(ValueError):
        create_preconditioner(well_conditioned_matrix, config)


def test_factory_requires_neural_config_for_neural_type(
    well_conditioned_matrix: torch.Tensor,
) -> None:
    """Factory requires NeuralPreconditionerConfig for neural type."""
    config = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": PreconditionerType.NEURAL})

    with pytest.raises(TypeError):
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
    matrix = torch.eye(2, dtype=torch.float64)
    result = create_scheduled_preconditioner(primary, schedule, matrix=matrix)

    assert isinstance(result, ScheduledPreconditioner)
    assert result._limit_iters == 10
    assert result._start_iter == 0


def test_create_scheduled_preconditioner_with_start_iter() -> None:
    """Builder forwards delayed activation to ScheduledPreconditioner."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(start_iter=7, limit_iters=10)
    matrix = torch.eye(2, dtype=torch.float64)
    result = create_scheduled_preconditioner(primary, schedule, matrix=matrix)

    assert isinstance(result, ScheduledPreconditioner)
    assert result._start_iter == 7
    assert result._limit_iters == 10


def test_create_scheduled_preconditioner_wraps_unlimited_delayed_schedule() -> None:
    """Delayed unlimited schedules are still wrapped."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(start_iter=3, limit_iters=-1)
    matrix = torch.eye(2, dtype=torch.float64)
    result = create_scheduled_preconditioner(primary, schedule, matrix=matrix)

    assert isinstance(result, ScheduledPreconditioner)
    assert result._start_iter == 3
    assert result._limit_iters is None


def test_create_scheduled_preconditioner_delayed_schedule_uses_fallback_before_start(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Delayed factory schedules use the identity fallback before activation."""
    primary = JacobiPreconditioner().setup(well_conditioned_matrix)
    schedule = PreconditionerScheduleConfig(start_iter=2, limit_iters=-1)
    result = create_scheduled_preconditioner(primary, schedule, matrix=well_conditioned_matrix)

    early_ctx = PreconditionerContext(iteration=1, residual_norm=1.0, rhs_norm=1.0)
    active_ctx = PreconditionerContext(iteration=2, residual_norm=1.0, rhs_norm=1.0)

    assert_close(result.apply(residual_vector, early_ctx), residual_vector)
    assert_close(
        result.apply(residual_vector, active_ctx),
        torch.tensor([0.25, 2 / 3, 1.5, 2.0], dtype=torch.float64),
    )


def test_scheduled_wrapper_does_not_setup_an_already_ready_primary(
    monkeypatch: pytest.MonkeyPatch,
    well_conditioned_matrix: torch.Tensor,
) -> None:
    """A scheduled build owns exactly one setup of a matrix-dependent primary."""
    primary = JacobiPreconditioner()
    calls = 0
    original_setup = primary.setup

    def setup(matrix: torch.Tensor, *args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_setup(matrix, *args, **kwargs)

    monkeypatch.setattr(primary, "setup", setup)
    primary.setup(well_conditioned_matrix)
    create_scheduled_preconditioner(
        primary,
        PreconditionerScheduleConfig(limit_iters=2),
        matrix=well_conditioned_matrix,
    )

    assert calls == 1


def test_create_scheduled_preconditioner_default_fallback() -> None:
    """Identity is used as the default fallback."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=5)
    matrix = torch.eye(2, dtype=torch.float64)
    result = create_scheduled_preconditioner(primary, schedule, matrix=matrix)

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
    matrix = torch.eye(2, dtype=torch.float64)
    result = create_scheduled_preconditioner(primary, schedule, matrix=matrix)

    ctx = PreconditionerContext(iteration=10, residual_norm=1.0, rhs_norm=1.0)
    r = torch.tensor([1.0, 2.0], dtype=torch.float64)
    z = result.apply(r, ctx)

    assert_close(z, r)
