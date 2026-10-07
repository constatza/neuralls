"""CSR input for SmootherFilteredProbesStrategy must match the dense result.

The strategy's damping sweep runs on the matrix in whichever format it
arrives; these tests pin that the sparse path reproduces the dense output
for the same values, and never densifies the CSR input.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.domain.generation import run_generation

SPD_DIM = 12
SPD_SEED = 3
PROBE_SEED = 11
STRATEGY = "smoother_filtered_probes"


@pytest.fixture
def dense_spd() -> np.ndarray:
    """Seeded dense SPD matrix, diagonally shifted so it stays well conditioned."""
    gen = np.random.default_rng(SPD_SEED)
    base = gen.standard_normal((SPD_DIM, SPD_DIM))
    return base.T @ base + np.eye(SPD_DIM)


@pytest.fixture
def csr_spd(dense_spd: np.ndarray) -> csr_array:
    """CSR view of ``dense_spd`` with the same values (zeros dropped by scipy)."""
    return csr_array(dense_spd)


@pytest.fixture
def toarray_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Count calls to ``csr_array.toarray``; the sparse path must make none."""
    calls: list[int] = []
    original: Callable[..., Any] = csr_array.toarray

    def counting_toarray(self: csr_array, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(csr_array, "toarray", counting_toarray)
    yield calls


def _cfg(omega: float | None, distribution: str = "gaussian") -> dict[str, Any]:
    """Strategy config shared by the dense and CSR runs."""
    cfg: dict[str, Any] = {
        "samples": 6,
        "seed": PROBE_SEED,
        "stop": 4,
        "start": 1,
        "probe_distribution": distribution,
    }
    if omega is not None:
        cfg["omega"] = omega
    return cfg


@pytest.mark.parametrize("omega", [0.67, None])
@pytest.mark.parametrize("distribution", ["gaussian", "rademacher"])
def test_csr_output_matches_dense(
    dense_spd: np.ndarray,
    csr_spd: csr_array,
    omega: float | None,
    distribution: str,
) -> None:
    """CSR and dense inputs with identical values produce the same traces."""
    cfg = _cfg(omega, distribution)

    dense = run_generation(STRATEGY, dense_spd, cfg=cfg).residual_traces
    sparse = run_generation(STRATEGY, csr_spd, cfg=cfg).residual_traces

    assert dense is not None and sparse is not None
    assert dense.solutions.shape == sparse.solutions.shape
    np.testing.assert_allclose(sparse.solutions, dense.solutions, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sparse.residuals, dense.residuals, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(sparse.sample_indices, dense.sample_indices)
    np.testing.assert_array_equal(sparse.iteration_indices, dense.iteration_indices)


def test_csr_path_never_densifies(csr_spd: csr_array, toarray_calls: list[int]) -> None:
    """Running on a CSR matrix must not call ``toarray`` on it."""
    run_generation(STRATEGY, csr_spd, cfg=_cfg(omega=0.67))

    assert toarray_calls == []
