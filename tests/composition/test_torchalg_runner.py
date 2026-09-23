"""Regression test for the no-autograd guard around torchalg's PCG solve.

torchalg never wraps its own solve loop in ``torch.no_grad()``/
``torch.inference_mode()``. Dataset generation calls ``run_traced_pcg``
thousands of times per binding (``gaussian_residuals``/``residuals`` run one
call per sample), so without this guard every call builds a full, unused
autograd graph purely to be discarded by the ``.detach()`` at the end — see
``neuralls/composition/solvers/torchalg_runner.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from neuralls.composition.solvers.torchalg_runner import run_traced_pcg


def test_run_traced_pcg_disables_grad_tracking(
    spd_system: tuple[np.ndarray, np.ndarray, np.ndarray],
    monkeypatch,
) -> None:
    """``run_traced_pcg`` must call torchalg's ``pcg`` with grad tracking off (via ``inference_mode``)."""
    matrix, rhs, initial_guess = spd_system
    grad_enabled_during_solve: list[bool] = []

    def _fake_pcg(
        a_tensor: torch.Tensor, b_tensor: torch.Tensor, x0_tensor: torch.Tensor, **_kwargs
    ):
        grad_enabled_during_solve.append(torch.is_grad_enabled())
        return x0_tensor, object()

    monkeypatch.setattr("neuralls.composition.solvers.torchalg_runner.pcg", _fake_pcg)

    run_traced_pcg(matrix, rhs, initial_guess, maxiter=5, rtol=1e-6, atol=1e-12)

    assert grad_enabled_during_solve == [False]
