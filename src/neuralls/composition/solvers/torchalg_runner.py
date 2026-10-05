"""Adapters that map neuralls workflow inputs to torchalg solver calls."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_array
from torchalg import pcg
from torchalg.models.result import SolverResult
from torchalg.monitoring import TraceMode

from neuralls.shared.types import SystemMatrix

_TORCH_VALUE_DTYPE = torch.float64
"""Floating dtype of every operator and vector handed to torchalg."""

_TORCH_INDEX_DTYPE = torch.int64
"""Index dtype of CSR ``indptr``/``indices`` handed to torchalg."""


def _to_torch_operator(matrix: SystemMatrix) -> torch.Tensor:
    """Convert a workflow system matrix into a float64 torch operator.

    Dense input becomes a dense tensor. CSR input becomes one
    ``torch.sparse_csr_tensor`` built from the scipy buffers without
    densifying, so the O(n^2) memory cost never enters the solve path.

    Args:
        matrix: System matrix in either supported format.

    Returns:
        A dense or CSR float64 tensor of shape ``(n, n)``.
    """
    match matrix:
        case np.ndarray():
            return torch.as_tensor(matrix, dtype=_TORCH_VALUE_DTYPE)
        case csr_array():
            return torch.sparse_csr_tensor(
                torch.as_tensor(matrix.indptr, dtype=_TORCH_INDEX_DTYPE),
                torch.as_tensor(matrix.indices, dtype=_TORCH_INDEX_DTYPE),
                torch.as_tensor(matrix.data, dtype=_TORCH_VALUE_DTYPE),
                size=matrix.shape,
            )


def run_traced_pcg(
    A: SystemMatrix,
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

    The operator ``A`` may be dense or CSR; a CSR operator is converted once
    into a sparse CSR tensor and never densified.

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
            _to_torch_operator(A),
            torch.as_tensor(b, dtype=_TORCH_VALUE_DTYPE),
            torch.as_tensor(x0, dtype=_TORCH_VALUE_DTYPE),
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
            trace_mode=TraceMode.FULL,
        )
    return x.detach().cpu().numpy(), info
