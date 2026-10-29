"""Format-keyed preconditioner factory and neural model-input contract.

Covers the (type, format) dispatch table in
`neuralls.composition.preconditioners.factory`, the explicit logged densify
for dense-only neural types, and the model-input conversion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from scipy.sparse import csr_array, diags, identity, kron
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import (
    Identity,
    JacobiPreconditioner,
    NeuralPreconditioner,
)
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter
from torchalg.sparse.preconditioners.jacobi import JacobiPreconditioner as SparseJacobi

from neuralls.composition.preconditioners.builders import _lookup_builder
from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.composition.solvers.torchalg_runner import _to_torch_operator
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.dlkit.predictor_adapter import _prepare_model_input
from neuralls.shared.types import MatrixFormat

pytestmark = pytest.mark.deferred_io("config-loading")

RELATIVE_REMAINING_BOUND = 0.5
POD_RANK = 8
POD_RELATIVE_REMAINING_BOUND = 0.5
"""Largest allowed ratio ||A z - r|| / ||r|| after one CSR AMG application."""

GRID_SIDE = 8
"""Side of the 2D Laplacian fixture; the system has GRID_SIDE**2 = 64 unknowns."""


class _StubPredictor(ExtraInputPredictorPort):
    """Predictor that returns its input unchanged; never touches a checkpoint."""

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Return the residual unchanged."""
        del extra_inputs
        return residual

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """The stub needs no extra inputs beyond the residual."""
        return ()

    def cleanup(self) -> None:
        """Nothing to release."""


class _StubAdapter(PredictorAdapter):
    """Adapter that hands out `_StubPredictor` for any checkpoint."""

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Return a stub predictor regardless of the checkpoint arguments."""
        del checkpoint_path, config_path, data_config_path
        return _StubPredictor()


@pytest.fixture
def laplacian_csr() -> csr_array:
    """Symmetric positive-definite 2D 5-point Laplacian on a GRID_SIDE x GRID_SIDE grid."""
    one_d = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(GRID_SIDE, GRID_SIDE))
    eye = identity(GRID_SIDE, format="csr")
    laplacian = kron(eye, one_d) + kron(one_d, eye)
    return csr_array(laplacian)


@pytest.fixture
def laplacian_dense(laplacian_csr: csr_array) -> torch.Tensor:
    """Dense float64 copy of the Laplacian fixture."""
    return torch.as_tensor(laplacian_csr.toarray(), dtype=torch.float64)


@pytest.fixture
def stub_adapter() -> PredictorAdapter:
    """Adapter that never loads a real checkpoint."""
    return _StubAdapter()


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Existing placeholder checkpoint file for neural config validation."""
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"placeholder")
    return path


def test_csr_jacobi_request_returns_torchalg_sparse_class(laplacian_csr: csr_array) -> None:
    """A CSR Jacobi request resolves to torchalg's sparse `JacobiPreconditioner`."""
    config = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)

    precond = create_preconditioner(
        _to_torch_operator(laplacian_csr), config, matrix_format=MatrixFormat.CSR
    )

    assert isinstance(precond, SparseJacobi)


def test_dense_jacobi_request_returns_dense_class(laplacian_dense: torch.Tensor) -> None:
    """A dense Jacobi request resolves to the dense `JacobiPreconditioner`."""
    config = StandardPreconditionerConfig(type=PreconditionerType.JACOBI)

    precond = create_preconditioner(laplacian_dense, config)

    assert isinstance(precond, JacobiPreconditioner)
    assert not isinstance(precond, SparseJacobi)


@pytest.mark.parametrize("matrix_format", [MatrixFormat.DENSE, MatrixFormat.CSR])
def test_identity_resolves_for_both_formats(
    laplacian_csr: csr_array,
    laplacian_dense: torch.Tensor,
    matrix_format: MatrixFormat,
) -> None:
    """IDENTITY is format-free and returns `Identity` for either format."""
    config = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)
    matrix = (
        laplacian_dense
        if matrix_format is MatrixFormat.DENSE
        else _to_torch_operator(laplacian_csr)
    )

    precond: Preconditioner = create_preconditioner(matrix, config, matrix_format=matrix_format)

    assert isinstance(precond, Identity)


def test_neural_csr_request_reaches_dense_neural_builder(
    laplacian_csr: csr_array,
    mock_checkpoint: Path,
    stub_adapter: PredictorAdapter,
) -> None:
    """A CSR neural request reaches the dense neural builder."""
    config = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(
        _to_torch_operator(laplacian_csr),
        config,
        adapter=stub_adapter,
        matrix_format=MatrixFormat.CSR,
    )

    assert isinstance(precond, NeuralPreconditioner)


def test_missing_type_format_pair_raises_value_error() -> None:
    """A (type, format) pair with no registered builder raises."""
    with pytest.raises(ValueError):
        _lookup_builder(PreconditionerType.NEURAL, MatrixFormat.CSR)


def test_prepare_model_input_returns_float64_batched_tensor() -> None:
    """The model input is float64 with a leading batch axis."""
    value = torch.arange(6, dtype=torch.float32)

    tensor = _prepare_model_input(value, "cpu")

    assert tensor.dtype == torch.float64
    assert tuple(tensor.shape) == (1, 6)


def test_dense_fitted_checkpoint_is_rejected_for_a_csr_matrix(
    laplacian_csr: csr_array, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint that rebuilds as the dense POD class cannot back a sparse hierarchy."""
    from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy
    from torchalg.sparse.preconditioners.pod.coarsening import (
        PODCoarseningStrategy as SparsePODCoarseningStrategy,
    )

    from neuralls.composition.preconditioners import coarsening

    dense_fitted = PODCoarseningStrategy(rank=POD_RANK)
    monkeypatch.setattr(coarsening.torch, "load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(coarsening, "build_model_from_checkpoint", lambda _raw: dense_fitted)
    checkpoint = tmp_path / "pod.ckpt"
    checkpoint.write_bytes(b"placeholder")

    with pytest.raises(TypeError):
        coarsening._load_fitted_pod_coarsening(
            checkpoint,
            _to_torch_operator(laplacian_csr),
            SparsePODCoarseningStrategy,
        )
