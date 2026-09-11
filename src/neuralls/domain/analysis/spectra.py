"""Condition-number diagnostics for matrices and preconditioners.

``compute_condition_numbers`` estimates the 2-norm condition number of the
preconditioned operator ``M^-1 A`` via ARPACK's Implicitly Restarted
Arnoldi Method (Lehoucq, Sorensen & Yang, "ARPACK Users' Guide", SIAM,
1998 - ``scipy.sparse.linalg.eigs``), matrix-free (never materializes the
dense preconditioned operator).

This replaces two prior implementations, in order:

1. A hand-rolled (shifted) power-iteration estimator, found to
   catastrophically fail on ill-conditioned matrices with a clustered
   eigenvalue spectrum (e.g. a stiff body with a much-softer embedded
   region): plain power iteration's convergence rate is close to 1 for a
   clustered spectrum, and the shift subtraction used to recover the
   smallest eigenvalue (``shift - power_iterate(...)``) cancels almost all
   significant digits exactly when the matrix is most ill-conditioned - it
   understated the true condition number by several orders of magnitude,
   and its own defensive ``if lambda_min <= 0`` guard shows the authors
   knew it could go negative.
2. An exact dense delegation to
   ``torchalg.utils.spectral.preconditioned_condition_number`` (full
   ``torch.linalg.eigvals`` on the densely-materialized operator). Correct,
   but O(n^3) and not matrix-free - a regression against this module's own
   prior, deliberate matrix-free/GPU-capable design goal for large
   matrices. Several from-scratch matrix-free Lanczos variants (plain
   CG-Lanczos via the classical alpha/beta-to-tridiagonal connection, and
   two different from-scratch reorthogonalization schemes on top of it)
   were tried and rejected: plain CG-Lanczos was inconsistently accurate
   (up to ~60x off on some clustered spectra, from loss of orthogonality),
   and both custom reorthogonalization attempts introduced subtle bugs
   (one produced a negative eigenvalue estimate again, the other diverged
   to ~1e301) by invalidating the exact algebraic identity the alpha/beta
   tridiagonal-matrix construction depends on. Cross-checked directly
   against ARPACK on the same ill-conditioned test matrices during
   development: ARPACK matched the exact eigenvalue-ratio condition number
   to 3-4 significant figures on every case, including the worst one that
   was ~60x off under plain CG-Lanczos - implicit restarting is doing real,
   necessary numerical work here, not just a speed optimization, and is
   not something to casually reimplement from scratch under time pressure.

``M^-1 A`` is generally non-symmetric as a stored/applied operator (even
for SPD ``M``, ``A``), so ``scipy.sparse.linalg.eigsh`` (which assumes
symmetry in the standard inner product) does not apply directly without
first symmetrizing via ``M^-1/2`` - which is unavailable when only ``M^-1``
is given as an opaque callable. ``eigs`` (general Arnoldi, no symmetry
assumption) sidesteps this entirely and was verified to return effectively
real eigenvalues (negligible imaginary parts) for this SPD-preconditioned-SPD
case, as expected since ``M^-1 A`` is similar to the symmetric
``M^-1/2 A M^-1/2``.

3. (This chapter) ARPACK's plain ``which="SM"`` Arnoldi, unmodified, was
   found to fail to converge (``ArpackNoConvergence``) on some AMG-family
   preconditioners against realistically large, ill-conditioned matrices -
   the textbook explanation is that a Krylov subspace naturally resolves
   *large*-magnitude Ritz values first, so the smallest eigenvalue of a wide
   or clustered spectrum is the hardest one for plain Arnoldi to isolate
   without a spectral transform (Lehoucq, Sorensen & Yang, ARPACK Users'
   Guide, ch. 4 - shift-invert mode is the standard remedy). Shift-invert
   was investigated and rejected here: it requires a direct factorization
   of ``M^-1 A - sigma I`` (or an equivalent generalized-problem
   factorization), which in turn requires either materializing the dense
   ``M^-1 A`` operator (as expensive as the exact dense fallback this
   module already has, so no gain) or forward-applying ``M`` (which, like
   the ``eigsh`` case above, an opaque ``M^-1``-only preconditioner callable
   cannot provide - and AMG specifically has no explicit forward ``M`` to
   begin with, only an implicit V-cycle action). An *inexact* shift-invert
   (solving ``(M^-1 A - sigma I) x = v`` iteratively instead of via
   factorization) was also rejected: that inner solve is exactly as
   ill-conditioned as the outer problem being diagnosed, so it buys no
   convergence improvement - it's circular, not a fix.
   The actual fix applied instead, entirely within ARPACK's own documented
   usage, needed no new algorithm:
   (a) ``_DENSE_FALLBACK_MAX_DIMENSION`` widens the exact dense fallback
   from ARPACK's bare ``k < n - 1`` API constraint (``n >= 3``) to a real,
   measured cost budget (dense ``eigvals`` measured at ~1.1s for n=2000 on
   this project's hardware) - covering smaller comparison-run matrix sizes
   with an algorithm that cannot fail to converge, at negligible cost.
   (b) For matrices above that budget, ``eigs`` is now called with an
   explicit, more generous ``ncv`` (Krylov subspace size) than scipy's
   default (``min(n, 20)`` for ``k=1``) - too small a basis is the
   documented, standard explanation for poor Arnoldi convergence on spectra
   with many clustered eigenvalues between the extremes being sought.
   (c) If ARPACK still raises ``ArpackNoConvergence`` despite (b), that is
   now caught and the exact dense method is used as a last-resort fallback
   for that one preconditioner, rather than silently returning NaN - the
   diagnostic now trades speed for correctness only in the rare case it's
   actually needed, instead of failing outright.
   ``_ARPACK_NCV = 100`` was cross-checked against a *fifth* matrix-free
   attempt (a pure-torch, scipy-free thick-restart/Krylov-Schur Arnoldi,
   explored in ``torchalg`` and ultimately not adopted - see its
   ``.claude/plan.md`` for why) as part of that investigation: ``ncv=100``
   converged to the correct value on every clustered/near-singular/extreme
   (up to ~1e12) condition-number construction tried, from n=300 to
   n=10000, except a deliberately pathological worst case (an exact 50/50
   eigenvalue-count split between two widely-separated clusters, not
   representative of real FEM/AMG spectra) where it was merely very slow,
   not wrong - that case still resolves correctly via the dense fallback in
   (c) if it ever fails to converge in practice.

ARPACK's ``eigs`` requires ``k < n - 1`` (its own documented constraint);
for ``k=1`` that means ``n >= 3`` - trivially covered by
``_DENSE_FALLBACK_MAX_DIMENSION`` being far larger than 3.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs
from torchalg.utils.spectral import preconditioned_condition_number

PreconditionerCallable = Callable[[torch.Tensor], torch.Tensor]

_DENSE_FALLBACK_MAX_DIMENSION = 2000
"""Matrices this size or smaller use the exact dense method unconditionally.

Measured ~1.1s for a dense ``torch.linalg.eigvals`` at n=2000 on this
project's hardware - cheap enough, for a once-per-comparison diagnostic, to
prefer an algorithm that cannot fail to converge over ARPACK's matrix-free
estimate. Also subsumes ARPACK's own ``k < n - 1`` API constraint.
"""

_ARPACK_NCV = 100
"""Krylov subspace size passed to ``eigs`` above the dense-fallback budget.

scipy's default for ``k=1`` is ``min(n, 20)`` - too small to reliably
resolve an extreme eigenvalue on a wide or clustered spectrum (ARPACK
Users' Guide's standard remedy for poor convergence is a larger ``ncv``).
"""


def compute_condition_numbers(
    matrix: np.ndarray,
    preconditioners: dict[str, PreconditionerCallable],
) -> dict[str, float]:
    """Estimate the 2-norm condition number of each preconditioned system.

    Args:
        matrix: System matrix A, shape ``(n, n)``.
        preconditioners: Preconditioner callables keyed by name.

    Returns:
        Condition number per preconditioner name (NaN on failure - e.g. a
        preconditioner callable that raises, or a preconditioned operator
        with a genuinely singular/near-singular spectrum).
    """
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float64)
    n = matrix_tensor.shape[0]
    cond_numbers: dict[str, float] = {}
    for name, preconditioner in preconditioners.items():
        try:
            if n <= _DENSE_FALLBACK_MAX_DIMENSION:
                cond_numbers[name] = preconditioned_condition_number(matrix_tensor, preconditioner)
            else:
                cond_numbers[name] = _arnoldi_condition_number(
                    matrix, matrix_tensor, preconditioner
                )
        except (ValueError, RuntimeError, torch.linalg.LinAlgError) as exc:
            cond_numbers[name] = float("nan")
            logger.warning("Could not compute condition number for '{}': {}", name, exc)
    return cond_numbers


def _arnoldi_condition_number(
    matrix: np.ndarray,
    matrix_tensor: torch.Tensor,
    preconditioner: PreconditionerCallable,
) -> float:
    """Matrix-free condition number of ``M^-1 A`` via ARPACK's Arnoldi method.

    Args:
        matrix: System matrix A as numpy, shape ``(n, n)``.
        matrix_tensor: The same matrix as a float64 torch tensor (shares
            memory-equivalent data, kept in sync by the caller), passed to
            ``preconditioner`` so torch-native preconditioners never see a
            numpy array.
        preconditioner: Applies ``M^-1`` to a length-``n`` torch tensor.

    Falls back to the exact dense method if ARPACK fails to converge even
    with a generous Krylov subspace (``_ARPACK_NCV``) - see this module's
    docstring, chapter 3, for why that's preferred over a matrix-free
    algorithm change.

    Returns:
        ``max(|eig|) / min(|eig|)`` of ``M^-1 A``, from the two extreme
        eigenvalues ARPACK converges (largest-magnitude, smallest-magnitude),
        or from the exact dense fallback if ARPACK doesn't converge.

    Raises:
        ValueError: If the resolved smallest-magnitude eigenvalue is
            non-positive (a genuinely singular/near-singular or non-SPD
            preconditioned operator).
    """

    def matvec(vector: np.ndarray) -> np.ndarray:
        preconditioned = preconditioner(
            matrix_tensor @ torch.as_tensor(vector, dtype=torch.float64)
        )
        return preconditioned.detach().cpu().numpy()

    operator = LinearOperator(matrix.shape, matvec=matvec, dtype=np.float64)
    ncv = min(matrix.shape[0] - 1, _ARPACK_NCV)
    try:
        lambda_max = complex(eigs(operator, k=1, which="LM", ncv=ncv, return_eigenvectors=False)[0])
        lambda_min = complex(eigs(operator, k=1, which="SM", ncv=ncv, return_eigenvectors=False)[0])
    except ArpackNoConvergence as exc:
        logger.warning(
            "ARPACK failed to converge with ncv={} on a {}x{} operator; "
            "falling back to the exact dense method: {}",
            ncv,
            *matrix.shape,
            exc,
        )
        return preconditioned_condition_number(matrix_tensor, preconditioner)

    lambda_max_real = lambda_max.real
    lambda_min_real = lambda_min.real
    if lambda_min_real <= 0:
        raise ValueError(f"non-positive smallest eigenvalue estimate: {lambda_min_real}")
    return lambda_max_real / lambda_min_real


def format_scientific(value: float, sig_figs: int = 4) -> str:
    return f"{value:.{sig_figs}e}"


def plot_condition_numbers(
    cond_numbers: dict[str, float],
    *,
    save_dir: Path | None = None,
    suffix: str = "conditions",
    title: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> Path | None:
    """Plot condition numbers for preconditioners as horizontal bar chart.

    Args:
        cond_numbers: Dictionary mapping preconditioner names to condition numbers
        save_dir: Directory to save the plot
        suffix: Suffix for the output filename
        title: Optional title for the plot
        rtol: Optional relative tolerance parameter to display
        atol: Optional absolute tolerance parameter to display

    Returns:
        Path to saved plot, or None if no condition numbers provided
    """
    if not cond_numbers:
        return None
    labels = list(cond_numbers.keys())
    values = [cond_numbers[name] for name in labels]

    # Dynamic figsize based on number of labels
    figsize = (9, max(3, len(labels) * 0.6))
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(labels, values)
    ax.set_xscale("log")
    ax.set_xlabel("Condition number (λ_max/λ_min)")

    # Build subtitle from non-None parameters
    subtitle_parts = []
    if rtol is not None:
        subtitle_parts.append(f"rtol={rtol:.0e}")
    if atol is not None:
        subtitle_parts.append(f"atol={atol:.0e}")

    if subtitle_parts:
        ax.set_title(", ".join(subtitle_parts), fontsize=9)

    fig.suptitle(title or "Condition Numbers by Preconditioner", fontsize=13, fontweight="bold")

    # Annotations on the right side of bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2.0,
            format_scientific(value, sig_figs=4),
            ha="left",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    cond_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cond_path = Path(save_dir) / f"preconditioner_condition_numbers_{suffix}.png"
        fig.savefig(cond_path, dpi=150)
        logger.info(f"Saved condition number plot to: {cond_path}")
    else:
        plt.show()
    plt.close(fig)
    return cond_path
