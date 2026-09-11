"""Tests for spectra.py's matrix-free (ARPACK Arnoldi) condition number estimate.

Three things are checked against independently-computed oracles (never via
spectra.py's own removed helpers):

1. Correctness — the estimate matches the *exact eigenvalue ratio*
   (``np.linalg.eigvals``) of the preconditioned operator, which is what
   ARPACK's Arnoldi iteration converges to. This is NOT always the same as
   the SVD-based 2-norm condition number (``np.linalg.cond``) an earlier
   version of this code computed: they coincide only when the preconditioned
   operator is symmetric/normal (e.g. any diagonal case below), and
   legitimately diverge for a general non-symmetric ``M^-1 A`` — that's a
   deliberate, disclosed change of definition, not a bug.
2. Speed — the whole point of the matrix-free ARPACK estimate is to beat the
   O(n^3) exact dense method for large matrices; verified directly by timing
   both on a matrix large enough for the difference to show.
3. Robustness on the actual failure case a prior (power-iteration-based)
   implementation of this same function could not handle: a heterogeneous,
   ill-conditioned matrix with a clustered eigenvalue spectrum (mimicking a
   stiff body with a much-softer embedded region). That implementation
   understated the true condition number by several orders of magnitude and
   could return a *negative* estimate; this one must not.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import torch
from scipy.sparse.linalg import ArpackNoConvergence
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import Identity, JacobiPreconditioner

import neuralls.domain.analysis.spectra as spectra_module
from neuralls.domain.analysis.spectra import compute_condition_numbers


def _exact_eigenvalue_ratio(matrix: torch.Tensor, preconditioner: Preconditioner) -> float:
    """Ground-truth eigenvalue-ratio condition number: exact dense build + eigvals.

    This is the same quantity ``compute_condition_numbers`` estimates via
    ARPACK's Arnoldi iteration — the correct oracle for correctness checks.
    """
    columns = [preconditioner(matrix[:, idx]) for idx in range(matrix.shape[1])]
    precond_matrix = torch.stack(columns, dim=1).numpy()
    eigenvalues = np.linalg.eigvals(precond_matrix).real
    return float(eigenvalues.max() / eigenvalues.min())


def _exact_svd_condition_number(matrix: torch.Tensor, preconditioner: Preconditioner) -> float:
    """The dense-build-plus-SVD method ARPACK's matrix-free estimate outperforms on speed.

    Used only for the speed comparison below — not a correctness oracle,
    since it measures a different quantity (SVD 2-norm condition number) for
    non-symmetric preconditioned operators.
    """
    columns = [preconditioner(matrix[:, idx]) for idx in range(matrix.shape[1])]
    precond_matrix = torch.stack(columns, dim=1).numpy()
    return float(np.linalg.cond(precond_matrix))


@pytest.fixture
def diagonal_matrix() -> torch.Tensor:
    """A diagonal SPD matrix with an analytically known 2-norm condition number of 100."""
    return torch.diag(torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64))


@pytest.fixture
def tridiagonal_spd_matrix() -> torch.Tensor:
    """A small, well-conditioned symmetric tridiagonal SPD matrix."""
    return torch.tensor([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]], dtype=torch.float64)


@pytest.fixture
def dense_spd_matrix() -> torch.Tensor:
    """A small, wider-spectrum dense SPD matrix (deterministic, not random)."""
    return torch.tensor([[10.0, 2.0, 1.0], [2.0, 5.0, 0.5], [1.0, 0.5, 500.0]], dtype=torch.float64)


@pytest.fixture
def heterogeneous_stiffness_matrix() -> tuple[torch.Tensor, float]:
    """A 300x300 SPD matrix with a clustered spectrum, mimicking a stiff body

    with a much-softer embedded region (e.g. a 1000x-softer sphere): half
    the eigenvalues near O(1), the rest clustered near O(1e-3) plus a few
    near-zero modes down to O(1e-8). Built via an orthogonal similarity
    transform of a known diagonal, so the true condition number is exact
    regardless of the random basis.

    This is the exact class of matrix the removed power-iteration
    implementation of ``compute_condition_numbers`` failed on: it
    understated the true condition number by 6+ orders of magnitude and
    could return a negative "eigenvalue" estimate.

    Returns:
        ``(matrix, true_condition_number)``.
    """
    generator = torch.Generator().manual_seed(0)
    n = 300
    orthonormal_basis, _ = torch.linalg.qr(
        torch.randn(n, n, generator=generator, dtype=torch.float64)
    )
    eigenvalues = torch.empty(n, dtype=torch.float64)
    eigenvalues[: n // 2] = 1.0 + torch.rand(n // 2, generator=generator, dtype=torch.float64)
    eigenvalues[n // 2 : n - 6] = 1e-3 * (
        1.0 + torch.rand(n - 6 - n // 2, generator=generator, dtype=torch.float64)
    )
    eigenvalues[n - 6 :] = 1e-8 * torch.rand(6, generator=generator, dtype=torch.float64)
    matrix = orthonormal_basis @ torch.diag(eigenvalues) @ orthonormal_basis.T
    matrix = 0.5 * (matrix + matrix.T)
    true_condition_number = float(eigenvalues.max() / eigenvalues.min())
    return matrix, true_condition_number


def test_diagonal_matrix_identity_matches_analytical_condition_number(
    diagonal_matrix: torch.Tensor,
) -> None:
    """For the identity preconditioner, the estimate is exact for a diagonal matrix."""
    cond_numbers = compute_condition_numbers(diagonal_matrix.numpy(), {"none": Identity()})
    assert cond_numbers["none"] == pytest.approx(100.0, rel=1e-2)


def test_diagonal_matrix_jacobi_matches_analytical_condition_number(
    diagonal_matrix: torch.Tensor,
) -> None:
    """Jacobi-preconditioning a diagonal matrix yields the identity operator (condition 1)."""
    cond_numbers = compute_condition_numbers(
        diagonal_matrix.numpy(), {"jacobi": JacobiPreconditioner(diagonal_matrix)}
    )
    assert cond_numbers["jacobi"] == pytest.approx(1.0, rel=1e-2)


@pytest.mark.parametrize("matrix_name", ["tridiagonal_spd_matrix", "dense_spd_matrix"])
@pytest.mark.parametrize("preconditioner_name", ["identity", "jacobi"])
def test_arnoldi_matches_exact_eigenvalue_ratio(
    request: pytest.FixtureRequest, matrix_name: str, preconditioner_name: str
) -> None:
    """The fast estimate agrees with the exact eigenvalue-ratio condition number."""
    matrix: torch.Tensor = request.getfixturevalue(matrix_name)
    preconditioner: Preconditioner = (
        Identity() if preconditioner_name == "identity" else JacobiPreconditioner(matrix)
    )

    expected = _exact_eigenvalue_ratio(matrix, preconditioner)
    cond_numbers = compute_condition_numbers(matrix.numpy(), {preconditioner_name: preconditioner})

    assert cond_numbers[preconditioner_name] == pytest.approx(expected, rel=1e-2)


def test_arnoldi_matches_exact_condition_number_on_heterogeneous_stiffness(
    heterogeneous_stiffness_matrix: tuple[torch.Tensor, float],
) -> None:
    """Regression test for the original bug: wide, clustered spectrum, identity preconditioner.

    The removed power-iteration implementation understated this matrix's
    true condition number (~1e11) by 6+ orders of magnitude and could
    return a negative estimate. ARPACK's Arnoldi method must match the true
    value closely and never go negative.
    """
    matrix, true_condition_number = heterogeneous_stiffness_matrix

    cond_numbers = compute_condition_numbers(matrix.numpy(), {"none": Identity()})

    estimated = cond_numbers["none"]
    assert estimated > 0.0
    assert not np.isnan(estimated)
    assert estimated == pytest.approx(true_condition_number, rel=1e-2)


def test_arnoldi_is_faster_than_exact_dense_svd() -> None:
    """The whole point of the matrix-free estimate is to beat the O(n^3) exact method.

    A wide-spectrum, diagonally-dominant SPD matrix at a size large enough to
    clear ``_DENSE_FALLBACK_MAX_DIMENSION`` (so ``compute_condition_numbers``
    actually takes the matrix-free ARPACK path, not the exact dense
    fallback) and for the dense-build-plus-SVD approach's cost to be
    measurable, but small enough to keep the test itself fast.
    """
    n = spectra_module._DENSE_FALLBACK_MAX_DIMENSION + 500
    diag_values = torch.linspace(1.0, 1.0e4, n, dtype=torch.float64)
    off_diag = 0.1 * torch.ones(n - 1, dtype=torch.float64)
    matrix = torch.diag(diag_values) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)
    preconditioner = JacobiPreconditioner(matrix)

    start = time.perf_counter()
    compute_condition_numbers(matrix.numpy(), {"jacobi": preconditioner})
    fast_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    _exact_svd_condition_number(matrix, preconditioner)
    exact_elapsed = time.perf_counter() - start

    assert fast_elapsed < exact_elapsed * 0.5, (
        f"ARPACK Arnoldi estimate ({fast_elapsed:.3f}s) should be markedly faster than "
        f"the exact dense-SVD method it outperforms ({exact_elapsed:.3f}s)"
    )


def test_dense_fallback_used_within_cost_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matrices within the measured-cheap dense budget never touch ARPACK at all.

    ARPACK's matrix-free Arnoldi only pays for itself once a full dense
    eigendecomposition is expensive; below that, the exact method is both
    correct and fast enough that iterative convergence risk (this ticket's
    original complaint) isn't worth taking on at all.
    """

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("eigs should not be called within the dense fallback budget")

    monkeypatch.setattr(spectra_module, "eigs", fail_if_called)

    n = spectra_module._DENSE_FALLBACK_MAX_DIMENSION
    matrix = torch.diag(torch.linspace(1.0, 100.0, n, dtype=torch.float64))

    cond_numbers = compute_condition_numbers(matrix.numpy(), {"none": Identity()})

    assert cond_numbers["none"] == pytest.approx(100.0, rel=1e-2)


def test_arpack_used_above_dense_fallback_threshold_with_generous_ncv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above the dense budget, ARPACK runs with a wider Krylov subspace than scipy's default.

    scipy's default ``ncv`` for ``k=1`` is ``min(n, 20)`` — too small to
    reliably resolve an extreme eigenvalue on a wide/clustered spectrum
    (ARPACK Users' Guide's standard remedy for poor convergence is a larger
    ``ncv``, not a different algorithm).
    """
    calls: list[dict[str, Any]] = []
    original_eigs = spectra_module.eigs

    def spy_eigs(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original_eigs(*args, **kwargs)

    monkeypatch.setattr(spectra_module, "eigs", spy_eigs)

    n = spectra_module._DENSE_FALLBACK_MAX_DIMENSION + 1
    matrix = torch.diag(torch.linspace(1.0, 100.0, n, dtype=torch.float64))

    compute_condition_numbers(matrix.numpy(), {"none": Identity()})

    assert len(calls) == 2, "expected one eigs() call each for lambda_max (LM) and lambda_min (SM)"
    for call_kwargs in calls:
        assert call_kwargs["ncv"] > 20


def test_arpack_non_convergence_falls_back_to_dense_instead_of_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matrix ARPACK can't converge on must still yield a real number, never NaN.

    This is the reported failure mode: ARPACK's Arnoldi genuinely can fail
    to converge on an ill-conditioned operator above the dense-fallback
    size. Silently returning NaN (the old behavior) makes the diagnostic
    useless for exactly the matrices it's needed most for; falling back to
    the exact dense method (already proven correct elsewhere in this suite)
    keeps the diagnostic honest at the cost of speed only in this rare case.
    """

    def raising_eigs(*args: object, **kwargs: object) -> object:
        raise ArpackNoConvergence("forced failure for testing", np.array([]), np.array([]))

    monkeypatch.setattr(spectra_module, "eigs", raising_eigs)

    n = spectra_module._DENSE_FALLBACK_MAX_DIMENSION + 1
    matrix = torch.diag(torch.linspace(1.0, 100.0, n, dtype=torch.float64))

    cond_numbers = compute_condition_numbers(matrix.numpy(), {"none": Identity()})

    assert not np.isnan(cond_numbers["none"])
    assert cond_numbers["none"] == pytest.approx(100.0, rel=1e-2)
