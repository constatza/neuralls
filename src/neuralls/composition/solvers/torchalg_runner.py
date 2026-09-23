"""Adapters that map neuralls workflow inputs to torchalg solver calls."""

from __future__ import annotations

import numpy as np
import torch
from torchalg import pcg
from torchalg.models.result import SolverResult
from torchalg.monitoring import TraceMode


def run_traced_pcg(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    *,
    maxiter: int,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, SolverResult]:
    """Run torchalg PCG on NumPy-loaded workflow arrays and return trace data.

    Generation archives are still NumPy-backed. This function is the workflow
    boundary that converts those arrays once, runs torchalg with tensors, and
    converts only the solution back for the existing generation DTOs.
    ``SolverResult.direction_vectors`` is populated natively by torchalg
    under ``TraceMode.FULL``.

    torchalg never wraps its solve in ``torch.no_grad()``, so without this
    guard every call builds a full autograd graph purely to be discarded by
    the ``.detach()`` below — dataset generation calls this thousands of
    times per binding, and the accumulating graphs make Python's cyclic GC
    pauses long and unpredictable. ``inference_mode`` is used over
    ``no_grad`` since nothing downstream ever needs autograd metadata on
    these tensors, not even the version counters ``no_grad`` still tracks.
    """
    with torch.inference_mode():
        x, info = pcg(
            torch.as_tensor(A, dtype=torch.float64),
            torch.as_tensor(b, dtype=torch.float64),
            torch.as_tensor(x0, dtype=torch.float64),
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
            trace_mode=TraceMode.FULL,
        )
    return x.detach().cpu().numpy(), info
