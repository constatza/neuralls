"""Bridge from CSR components to a torch sparse CSR tensor."""

from __future__ import annotations

from typing import Final

import torch

from neuralls.platform.sparse_io.components import CsrComponents

TORCH_VALUE_DTYPE: Final = torch.float64
"""Floating dtype of tensors produced from CSR components."""

TORCH_INDEX_DTYPE: Final = torch.int64
"""Index dtype of CSR ``crow``/``col`` tensors produced from CSR components."""


def to_torch_csr(components: CsrComponents) -> torch.Tensor:
    """Build a float64 ``torch.sparse_csr`` tensor without densifying.

    Args:
        components: CSR components; ``indptr``/``indices`` must follow scipy's convention.

    Returns:
        A sparse CSR tensor of shape ``components.shape``.
    """
    return torch.sparse_csr_tensor(
        torch.as_tensor(components.indptr, dtype=TORCH_INDEX_DTYPE),
        torch.as_tensor(components.indices, dtype=TORCH_INDEX_DTYPE),
        torch.as_tensor(components.data, dtype=TORCH_VALUE_DTYPE),
        size=components.shape,
    )
