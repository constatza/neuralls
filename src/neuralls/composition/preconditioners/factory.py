"""Preconditioner factory for TOML workflow.

This module provides a simple factory function for creating preconditioners
from configuration objects. It lives in the assembly layer because it depends
on both the configuration layer (PreconditionerType, config models) and the
solver layer (concrete preconditioner classes).

For direct usage, just instantiate preconditioners directly:
    >>> precond = JacobiPreconditioner(matrix)

For TOML workflow:
    >>> config = load_comparison_config("comparison.toml")
    >>> precond = create_preconditioner(matrix, config.preconditioner)

Design:
    - The two functions here are the single construction path; the actual
      per-(type, format) construction logic lives in `builders.py`'s dispatch
      table, and coarsening-strategy construction lives in `coarsening.py`.
    - Supports dependency injection for neural preconditioners (testing)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchalg.preconditioners.base import Preconditioner

from neuralls.shared.types import MatrixFormat

from .builders import _BuildDeps, _lookup_builder, _resolve_format_input
from .coarsening import PredictorFactory

if TYPE_CHECKING:
    from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy
    from torchalg.preconditioners.ports import PredictorAdapter

    from neuralls.platform.config.models.preconditioner import ConcretePreconditionerConfig


def create_preconditioner(
    matrix: torch.Tensor,
    config: ConcretePreconditionerConfig,
    adapter: PredictorAdapter | None = None,
    inference_predictor_factory: PredictorFactory | None = None,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
) -> Preconditioner:
    """Create a preconditioner from configuration for a matrix in the given format.

    Args:
        matrix: System matrix A: a dense tensor for `MatrixFormat.DENSE`, a
            sparse CSR tensor for `MatrixFormat.CSR`.
        config: Preconditioner configuration from TOML
        adapter: Optional adapter for neural preconditioner (DI for testing)
        inference_predictor_factory: Optional batch-inference predictor
            factory for neural POD-2G coarsening (DI for testing); defaults
            to `create_inference_predictor` from `platform.dlkit.inference_adapter`.
        matrix_format: Storage format of `matrix`. Defaults to dense.

    Returns:
        Preconditioner instance

    Raises:
        ValueError: If no builder is registered for the (type, format) pair.
    """
    preconditioner, _ = create_preconditioner_with_coarsening(
        matrix, config, adapter, inference_predictor_factory, matrix_format
    )
    return preconditioner


def create_preconditioner_with_coarsening(
    matrix: torch.Tensor,
    config: ConcretePreconditionerConfig,
    adapter: PredictorAdapter | None = None,
    inference_predictor_factory: PredictorFactory | None = None,
    matrix_format: MatrixFormat = MatrixFormat.DENSE,
) -> tuple[Preconditioner, CoarseningStrategy | None]:
    """Create a preconditioner, also returning its coarsening strategy when it has one.

    This is the single construction path; `create_preconditioner` returns its first
    element. AMG's realized coarse dimension is only knowable from the coarsening
    strategy actually used to build the hierarchy, not from config alone (POD's
    ``rank`` is often an energy threshold; AMG's ``theta`` yields an emergent
    aggregate count). Diagnostics that need the realized dimension must reuse this
    coarsening object rather than fitting a second one.

    The sparse AMG preset builds its aggregation internally and does not expose a
    coarsening object, so for CSR AMG the coarsening is ``None``.

    Args:
        matrix: System matrix A, in the format named by `matrix_format`.
        config: Preconditioner configuration from TOML.
        adapter: Optional adapter for neural preconditioner (DI for testing).
        inference_predictor_factory: Optional batch-inference predictor
            factory for neural POD-2G coarsening (DI for testing).
        matrix_format: Storage format of `matrix`. Defaults to dense.

    Returns:
        The preconditioner, and its coarsening strategy if the dense AMG path
        built one (`None` for every other type and for sparse AMG).

    Raises:
        ValueError: If no builder is registered for the (type, format) pair.
        TypeError: If the config variant does not match its preconditioner type.
    """
    deps = _BuildDeps(adapter=adapter, inference_predictor_factory=inference_predictor_factory)
    operator, resolved_format = _resolve_format_input(matrix, config.type, matrix_format)
    builder = _lookup_builder(config.type, resolved_format)
    return builder(operator, config, deps)
